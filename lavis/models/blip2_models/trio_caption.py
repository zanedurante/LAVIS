"""
 Copyright (c) 2023, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import logging
import string
import random
import copy

import torch
import torch.nn as nn
from torch.cuda.amp import autocast as autocast
from transformers import T5TokenizerFast

from lavis.common.registry import registry
from lavis.models.blip2_models.blip2 import Blip2Base, disabled_train
from lavis.models.blip2_models.modeling_t5 import T5Config, T5ForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput
from timm.models.vision_transformer import PatchEmbed, Block
from lavis.models.base_model import BaseModel
from transformers import BertTokenizer
# TODO: Allow for better fp16 support since the model is frozen. Right now fp32 is used for the model.
# https://github.com/huggingface/transformers/issues/14189#issuecomment-961571628

from transformers import  AutoTokenizer, GPT2LMHeadModel,  OPTForCausalLM
import re

@registry.register_model("trio_t5")
class TrioT5(Blip2Base):
    """
    Trio T5 model.
    Supported model types:
        - flant5xl
        - flant5xxl
    Usage:
        >>> from lavis.models import load_model
        >>> model = load_model("trio_t5", "flant5xl")
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "flant5xl": "configs/models/blip2/blip2_instruct_flant5xl.yaml",
        "flant5xxl": "configs/models/blip2/blip2_instruct_flant5xxl.yaml",
    }

    def __init__(
        self,
        vit_model="eva_clip_g",
        img_size=224,
        drop_path_rate=0,
        use_grad_checkpoint=False,
        vit_precision="fp16",
        freeze_vit=False,
        num_query_token=32,
        t5_model="google/flan-t5-small",
        prompt="",
        max_txt_len=128,
        max_output_txt_len=256,
        apply_lemmatizer=False,
        num_few_shot_examples=0,
        few_shot_prob=0,
        qformer_text_input=True,
        num_frames=4,
    ):
        """
        apply_lemmatizer: when set to True, postprocess predict_answers() result with lemmas.
        """
        super().__init__()

        self.tokenizer = self.init_tokenizer(truncation_side="right")
        self.start_token_id = self.tokenizer.bos_token_id
        self.visual_encoder, self.ln_vision = self.init_vision_encoder(
            vit_model, img_size, drop_path_rate, use_grad_checkpoint, vit_precision, num_frames=num_frames
        )
        self.num_frames = num_frames
        # if freeze_vit:
            # for name, param in self.visual_encoder.named_parameters():
            #     param.requires_grad = False
            # self.visual_encoder = self.visual_encoder.eval()
            # self.visual_encoder.train = disabled_train
            # logging.info("freeze vision encoder")

        # self.Qformer, self.query_tokens = self.init_Qformer(
        #     num_query_token, self.visual_encoder.num_features
        # )

        # if not qformer_text_input:
        #     self.Qformer.bert.embeddings.word_embeddings = None
        #     self.Qformer.bert.embeddings.position_embeddings = None
        #     for layer in self.Qformer.bert.encoder.layer:
        #         layer.output = None
        #         layer.intermediate = None
        # else:
        #     self.Qformer.resize_token_embeddings(len(self.tokenizer))
        # self.Qformer.cls = None

        # self.Qformer, self.query_tokens = self.init_Qformer(
        #     num_query_token, self.visual_encoder.num_features
        # )
        
        # configuration = TransfoXLConfig()
        # configuration.d_embed = 1024
        # configuration.output_hidden_states=True
        self.model = OPTForCausalLM.from_pretrained("facebook/opt-125m") #TransfoXLLMHeadModel(configuration)
        self.model.resize_token_embeddings(len(self.tokenizer))
   
    def init_tokenizer(self, truncation_side="right"):
        NOOP_ACTION = {
            "ESC": 0,
            "back": 0,
            "drop": 0,
            "forward": 0,
            "hotbar.1": 0,
            "hotbar.2": 0,
            "hotbar.3": 0,
            "hotbar.4": 0,
            "hotbar.5": 0,
            "hotbar.6": 0,
            "hotbar.7": 0,
            "hotbar.8": 0,
            "hotbar.9": 0,
            "inventory": 0,
            "jump": 0,
            "left": 0,
            "right": 0,
            "sneak": 0,
            "sprint": 0,
            "swapHands": 0,
            "attack": 0,
            "use": 0,
            "pickItem": 0,
        }
        tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")
        new_tokens = list(NOOP_ACTION.keys())
        new_tokens = [ f'[{new_token}]' for new_token in new_tokens ]
        tokenizer.add_tokens(new_tokens)
        camera_actions = []
        for i in range(-49, 50):
            camera_actionx = f'[camerax{i}]'
            camera_actiony = f'[cameray{i}]'
            camera_actions.extend([camera_actionx, camera_actiony])
        
        tokenizer.add_tokens(camera_actions)
        tokenizer.add_tokens('[endofaction]')
        tokenizer.add_tokens('[startaction]')
        tokenizer.pad_token = tokenizer.eos_token
        # tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        # tokenizer.add_special_tokens({"bos_token": "[DEC]"})
        # tokenizer.pad_token = tokenizer.eos_token

        return tokenizer
    
    def generate(self, samples):
        print('start generate')
        if "image" in samples.keys():
            image = samples["image"]
        elif "video" in samples.keys():
            image = samples["video"]
        else:
            raise ValueError("No image or video input found in input dict.")
        
        text_output = samples["text_output"]
        pattern = r'(.*?\[endofaction\])'
        text_output_final = [ [s for s in re.split(pattern, text) if s] for text in text_output ]
        num_frames = self.num_frames

        actions = []
        
        preds = []
        masks = []
        print('num_frames: ', num_frames)
        with autocast(dtype=torch.float16):
            frames_embeddings = []
            for i in range(num_frames):
                features, mask, ids_restore =   self.visual_encoder(image[:, i, :, :, :].unsqueeze(1)) # (N, 1, C, H, W)
                image_embeds = self.ln_vision(features)
                pred = self.visual_encoder.forward_decoder(image_embeds, ids_restore)
                frames_embeddings.append(features)
                preds.append(pred)
                masks.append(mask)

            
            max_length = self.num_frames

            total_embeddings = torch.empty((features.shape[0], 0, 768), dtype=torch.float16).to(image.device)

            prediction_embeddings = torch.empty((features.shape[0], 0, 768), dtype=torch.float16).to(image.device)
            action = ""
            for idx in range(max_length):  
                
                text_output_idx = [ text[idx] for text in text_output_final ]
                
                # print('text_output_idx: ', text_output_idx)
                target_tokens = self.tokenizer(
                    text_output_idx,
                    padding='longest',
                    truncation=True,
                    max_length=self.num_frames * 10,
                    add_special_tokens = True,
                    # skip_special_tokens=False,
                    return_tensors="pt"
                ).to(image.device)

                input_ids = target_tokens.input_ids
            
                text_embeddings = self.model.model.get_input_embeddings()(input_ids)
                # image_embeddings = self.xl_proj(frames_embeddings[idx])
                image_embeddings = frames_embeddings[idx]
                # print(image_embeddings.shape)
                
                prediction_embeddings = torch.cat((total_embeddings, image_embeddings), dim = 1 )
    
                # Generate the next token
                # TODO: replace image with image special token in label
                for _ in range(10):
                    outputs = self.model(
                        inputs_embeds = prediction_embeddings,

                    )
                    next_token_logits =  outputs.logits[:, -1, :]
                    # print('next_token_logits: ', next_token_logits)
                    next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)

                    # print('next token: ', next_token)
                    # print('logits: ', torch.softmax(next_token_logits, dim=-1)[0, 50486])
                    result = self.tokenizer.decode(next_token[0], skip_special_tokens=False)
                    # print('result: ', result)
                    action += result
                    if result == '[endofaction]':
                        actions.append(action)
                        break

                    next_token_embedding = self.model.model.get_input_embeddings()(next_token)
                    prediction_embeddings = torch.cat((prediction_embeddings, next_token_embedding), dim = 1 )

                # print('=============')
                
            # print('++++++++++++++++++++++')
                total_embeddings = torch.cat((total_embeddings, image_embeddings, text_embeddings), dim = 1 )
        return {'preds': preds, 'actions': actions, 
                'image': image, 'action_gt': text_output, 'masks': masks}
        
    def forward(self, samples):

        # allow for image or video input
        if "image" in samples.keys():
            image = samples["image"]
        elif "video" in samples.keys():
            image = samples["video"]
        else:
            raise ValueError("No image or video input found in input dict.")
        
        # image = image.to(torch.float32)
        
        text_output = samples["text_output"]
        
        pattern = r'(.*?\[endofaction\])'
        text_output_final = [ [s for s in re.split(pattern, text) if s] for text in text_output ]

        num_frames = self.num_frames
        max_length = self.num_frames

        
        mae_loss = 0.0
        frames_embeddings = []
        b, t, c, h, w = image.shape
        video_batch = image.reshape(b * t, c, h, w)
        with autocast(dtype=torch.float16):
            features, mask, ids_restore =   self.visual_encoder(video_batch.unsqueeze(1)) # (N, 1, C, H, W)
            image_embeds = self.ln_vision(features)
            pred = self.visual_encoder.forward_decoder(image_embeds, ids_restore)
            mae_loss = mae_loss +  self.visual_encoder.forward_loss(video_batch.unsqueeze(1), pred, mask)

            image_embeds = image_embeds.reshape( b, self.num_frames, image_embeds.shape[-2], image_embeds.shape[-1])

            for i in range(num_frames):
                frames_embeddings.append(image_embeds[:, i, :, :])
            

            total_embeddings = torch.empty((b, 0, 768)).to(image.device)
            
            total_loss = 0.0
            labels = torch.empty((b, 0), dtype=torch.long).to(image.device)
            for idx in range(max_length):  
                
                text_output_idx = [ text[idx] for text in text_output_final ]
                
                # print('text_output_idx: ', text_output_idx)
                target_tokens = self.tokenizer(
                    text_output_idx,
                    padding='longest',
                    truncation=True,
                    max_length=self.num_frames * 10,
                    add_special_tokens = True,
                    # skip_special_tokens=False,
                    return_tensors="pt"
                ).to(image.device)

                input_ids = target_tokens.input_ids
            
                text_embeddings = self.model.model.get_input_embeddings()(input_ids)
                image_embeddings = frames_embeddings[idx]
                
                
                total_embeddings = torch.cat((total_embeddings, image_embeddings, text_embeddings), dim = 1 )
                text_labels = input_ids.clone()

                #prepend -100 to labels for image patches
                labels = torch.cat((labels, torch.ones((labels.shape[0], image_embeddings.shape[1]), dtype=torch.long).to(image.device) * -100, text_labels), dim=1)


                
                output = self.model(
                    inputs_embeds = total_embeddings,
                    labels = labels,
                )
                loss = output.loss
                total_loss += loss
            

        
            total_loss = total_loss + mae_loss
            return {"loss": total_loss, "pred": pred, "image": image, 'mask': mask}


    @classmethod
    def from_config(cls, cfg):
        print(cfg)
        vit_model = cfg.get("vit_model", "eva_clip_g_video")
        img_size = cfg.get("image_size")
        num_query_token = cfg.get("num_query_token")
        t5_model = cfg.get("t5_model")
        random_init_qformer = cfg.get("random_init_qformer", False)
        num_frames = cfg.get("num_frames", 4)

        drop_path_rate = cfg.get("drop_path_rate", 0)
        use_grad_checkpoint = cfg.get("use_grad_checkpoint", False)
        vit_precision = cfg.get("vit_precision", "fp16")
        freeze_vit = cfg.get("freeze_vit", True)

        prompt = cfg.get("prompt", "")
        max_txt_len = cfg.get("max_txt_len", 128)
        max_output_txt_len = cfg.get("max_output_txt_len", 256)

        apply_lemmatizer = cfg.get("apply_lemmatizer", False)

        num_few_shot_examples = cfg.get("num_few_shot_examples", 0)
        few_shot_prob = cfg.get("few_shot_prob", 0.0)

        qformer_text_input = cfg.get("qformer_text_input", True)

        model = cls(
            vit_model=vit_model,
            img_size=img_size,
            drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpoint,
            vit_precision=vit_precision,
            freeze_vit=freeze_vit,
            num_query_token=num_query_token,
            t5_model=t5_model,
            prompt=prompt,
            max_txt_len=max_txt_len,
            max_output_txt_len=max_output_txt_len,
            apply_lemmatizer=apply_lemmatizer,
            num_few_shot_examples=num_few_shot_examples,
            few_shot_prob=few_shot_prob,
            qformer_text_input=qformer_text_input,
            num_frames=num_frames,
        )

        # if qformer_text_input:
        #     # Hard-coded to load from BLIP-2 stage-1 pre-trained model (not ideal)
        #     model.load_from_pretrained(
        #         url_or_filename="https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_pretrained.pth"
        #     )
        if not random_init_qformer:
            model.load_checkpoint_from_config(cfg)

        return model
"""
@registry.register_model("trio_t5_video")
class TrioT5InstructVideo(Blip2T5Instruct):
    PRETRAINED_MODEL_CONFIG_DICT = {
        "flant5xl": "configs/models/blip2/blip2_instruct_flant5xl.yaml",
        "flant5xxl": "configs/models/blip2/blip2_instruct_flant5xxl.yaml",
    }

    def __init__(
        self,
        vit_model="eva_clip_g",
        img_size=224,
        drop_path_rate=0,
        use_grad_checkpoint=False,
        vit_precision="fp16",
        freeze_vit=True,
        num_query_token=32,
        t5_model="google/flan-t5-xl",
        prompt="",
        max_txt_len=128,
        max_output_txt_len=256,
        apply_lemmatizer=False,
        num_few_shot_examples=0,
        few_shot_prob=0,
        qformer_text_input=True,
    ):
        super().__init__(
            vit_model="eva_clip_g_video",
            img_size=224,
            drop_path_rate=0,
            use_grad_checkpoint=False,
            vit_precision="fp16",
            freeze_vit=True,
            num_query_token=32,
            t5_model=t5_model,
            prompt="",
            max_txt_len=128,
            max_output_txt_len=256,
            apply_lemmatizer=False,
            num_few_shot_examples=0,
            few_shot_prob=0,
            qformer_text_input=True,
        )
"""