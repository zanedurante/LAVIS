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

from transformers import  AutoTokenizer, AutoModelForCausalLM
import re
from lavis.models.model_utils import init_tokenizer
# from peft import get_peft_config, get_peft_model, LoraConfig, TaskType
# from huggingface_hub import login

@registry.register_model("robot_transformer")
class RobotTransformer(Blip2Base):
    """
    Trio T5 model.
    Supported model types:
        - flant5xl
        - flant5xxl
    Usage:
        >>> from lavis.models import load_model
        >>> model = load_model("robot_transformer", "flant5xl")
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
        mae_decoder = True,
    ):
        """
        apply_lemmatizer: when set to True, postprocess predict_answers() result with lemmas.
        """
        super().__init__()
        
        self.base_model_name = "facebook/opt-125m"

        self.tokenizer = init_tokenizer(self.base_model_name)
        self.start_token_id = self.tokenizer.bos_token_id
        self.visual_encoder, self.ln_vision = self.init_vision_encoder(
            vit_model, img_size, drop_path_rate, use_grad_checkpoint, vit_precision, num_frames=num_frames, mae_decoder=mae_decoder
        )

        self.model = AutoModelForCausalLM.from_pretrained(self.base_model_name ,
                                                           torch_dtype=torch.float32)
        self.model.resize_token_embeddings(len(self.tokenizer))
        # self.model = self.model.half()
        self.num_frames = num_frames

        # peft_config = LoraConfig(
        #             task_type="CAUSAL_LM",
        #             inference_mode=False, r=8, 
        #             lora_alpha=8, lora_dropout=0.1,
        #             target_modules=["q_proj", "v_proj"],
        #             modules_to_save=["embed_tokens", "lm_head"]
        # )


        # self.model = get_peft_model(self.model, peft_config)
        # self.model.print_trainable_parameters()

        self.model_size = 768
        self.linear_projection = nn.Linear(768, self.model_size).to(torch.float32)
     
   

    
    def generate(self, samples):
        image = samples["video"]
        instructions = samples["instructions"].to(image.device)

        actions_gt = samples["actions"].to(image.device)
        effector_translation = samples["effector_translation"].to(image.device)
        effector_target_translation = samples["effector_target_translation"].to(image.device)
        print(actions_gt)
        # exit('action gt')
        
        flat_tokens = [token for sublist in samples["actions"][0] for token in sublist]
        print(flat_tokens)
        decoded_string = self.tokenizer.decode(flat_tokens)
        print(decoded_string)

        max_length = image.shape[1]
        # print('start training')
        # print(samples.keys())
        
        instruction_embeds = self.model.model.get_input_embeddings()(instructions)
        # total_embeddings = torch.empty((instruction_embeds.shape[0], 0, 768), dtype=torch.bfloat16).to(image.device)
        total_embeddings = instruction_embeds #torch.cat((total_embeddings, instruction_embeds), dim = 1 )
        max_length = num_frames = self.num_frames

        preds = []
        actions = []
        masks = []
        actions_gt_return = []

        with autocast(dtype=torch.bfloat16):
            frames_embeddings = []
            b, t, c, h, w = image.shape
            video_batch = image.reshape(b * t, c, h, w)

            features, mask, ids_restore =   self.visual_encoder(video_batch.unsqueeze(1), use_mask = False)
            image_embeds = self.ln_vision(features)
            image_embeds = image_embeds.reshape( b, t, image_embeds.shape[-2], image_embeds.shape[-1])
            image_embeds = self.linear_projection(image_embeds)
            for i in range(t):
                frames_embeddings.append(image_embeds[:, i, :, :])
            
            # b, t, c, h, w = image.shape
            # video_batch = image.reshape(b * t, c, h, w)

            # features, mask, ids_restore =   self.visual_encoder(video_batch.unsqueeze(1)) # (N, 1, C, H, W)
            # image_embeds = self.ln_vision(features)
            # pred = self.visual_encoder.forward_decoder(image_embeds, ids_restore)
            # image_embeds = image_embeds.reshape( b, t, image_embeds.shape[-2], image_embeds.shape[-1])
            # for p in pred:
            #     preds.append(p)
            
            # for m in mask:
            #     masks.append(m)
            
            # for f in features:
            #     frames_embeddings.append(f)


            prediction_embeddings = instruction_embeds.clone()
            
            for idx in range(max_length):  
                action = "[STARTACTION]"
                image_embeddings = frames_embeddings[idx]
                action_input_ids = actions_gt[:, idx, :].clone()         
                action_embeddings = self.model.model.get_input_embeddings()(action_input_ids)

                result = self.tokenizer.batch_decode(action_input_ids, skip_special_tokens=False)
                actions_gt_return.append(result)
                
                # result = self.tokenizer.batch_decode(action_input_ids, skip_special_tokens=False)
                # print(result)

                effector_translation_input_ids = effector_translation[:, idx, :].clone()
                effector_translation_embeddings = self.model.model.get_input_embeddings()(effector_translation_input_ids)

                result = self.tokenizer.batch_decode(effector_translation_input_ids, skip_special_tokens=False)
                print(result)

                effector_target_translation_input_ids = effector_target_translation[:, idx, :].clone()
                effector_target_translation_embeddings = self.model.model.get_input_embeddings()(effector_target_translation_input_ids)

                result = self.tokenizer.batch_decode(effector_target_translation_input_ids, skip_special_tokens=False)
                print(result)
                
                prediction_embeddings = torch.cat((total_embeddings, image_embeddings,
                                                   effector_translation_embeddings, effector_target_translation_embeddings ), dim = 1 )
                # prediction_embeddings = torch.cat((total_embeddings, image_embeddings), dim = 1 )
    
                # Generate the next token
                # TODO: replace image with image special token in label
                for index in range(5):
                    if index == 0:
                        next_token = self.tokenizer.encode('[STARTACTION]')
                        next_token = torch.tensor(next_token).unsqueeze(0).to(image.device)
                        next_token_embedding = self.model.model.get_input_embeddings()(next_token)
                        prediction_embeddings = torch.cat((prediction_embeddings, next_token_embedding), dim = 1 )
                        continue

                    outputs = self.model(
                        inputs_embeds = prediction_embeddings,

                    )
                    next_token_logits =  outputs.logits[:, -1, :]
                    # print('next_token_logits: ', next_token_logits)
                    next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)

                    # print('next token: ', next_token)
                    # print('logits: ', torch.softmax(next_token_logits, dim=-1)[0, 50486])
                    result = self.tokenizer.decode(next_token[0], skip_special_tokens=False)
                    print('result: ', result)
                    action += result
                    if result == '[ENDOFACTION]':
                        actions.append(action)
                        break

                    next_token_embedding = self.model.model.get_input_embeddings()(next_token)
                    prediction_embeddings = torch.cat((prediction_embeddings, next_token_embedding), dim = 1 )

                print('=============')
                total_embeddings = torch.cat((total_embeddings, image_embeddings, effector_translation_embeddings,
                                              effector_target_translation_embeddings, action_embeddings), dim = 1 )
            print('++++++++++++++++++++++')
          
        return {'preds': preds, 'actions': actions, 
                'image': image, 'action_gt': actions_gt_return, 'masks': masks}
        
    def forward(self, samples):
        mae_loss = 0.0
        total_loss = 0.0

        loss1 = 0.0
        loss2 = 0.0
        loss3 = 0.0
        
        epoch = samples["epoch"]
        finetune = samples['finetune']

        if 'robot' in samples.keys():
            robot_samples = samples['robot']
        else:
            robot_samples = None

        if 'minecraft' in samples.keys():        
            minecraft_samples = samples['minecraft']
        else:
            minecraft_samples = None

        if 'calvin' in samples.keys():
            calvin_samples = samples['calvin']
        else:
            calvin_samples = None

        
     
        # processing robotics data 
        if robot_samples:
            image = robot_samples["video"].to(torch.float32)
            instructions = robot_samples["instructions"].to(image.device)

            actions = robot_samples["actions"].to(image.device)
            effector_translation = robot_samples["effector_translation"].to(image.device)
            effector_target_translation = robot_samples["effector_target_translation"].to(image.device)
            
            max_length = image.shape[1]
            # print('start training')
            # print(samples.keys())
        
        
            
            with autocast(dtype=torch.float16):
                instruction_embeds = self.model.model.get_input_embeddings()(instructions)
                # total_embeddings = torch.empty((instruction_embeds.shape[0], 0, self.model_size), dtype=torch.float16).to(image.device)
                total_embeddings = instruction_embeds.clone()
                
                frames_embeddings = []
                b, t, c, h, w = image.shape
                video_batch = image.reshape(b * t, c, h, w)

                # if epoch < 60 and self.training:
                features, mask, ids_restore =   self.visual_encoder(video_batch.unsqueeze(1), use_mask = True) # (N, 1, C, H, W)
                image_embeds = self.ln_vision(features)
                if not finetune:
                    pred = self.visual_encoder.forward_decoder(image_embeds, ids_restore)
                    mae_loss +=   self.visual_encoder.forward_loss(video_batch.unsqueeze(1), pred, mask)
                # else:
                #     features, mask, ids_restore =   self.visual_encoder(video_batch.unsqueeze(1), use_mask = False)
                #     image_embeds = self.ln_vision(features)
                #     pred = None
                
                image_embeds = image_embeds.reshape( b, t, image_embeds.shape[-2], image_embeds.shape[-1])
                image_embeds = self.linear_projection(image_embeds)

                for i in range(t):
                    frames_embeddings.append(image_embeds[:, i, :, :])
                

                # total_embeddings = torch.empty((b, 0, 768), dtype=torch.float16).to(image.device)
                
                
                labels = instructions.clone().to(image.device)
                for idx in range(max_length):  
                    action_input_ids = actions[:, idx, :].clone()         
                    action_embeddings = self.model.model.get_input_embeddings()(action_input_ids)
                    
                    effector_translation_input_ids = effector_translation[:, idx, :].clone()
                    effector_translation_embeddings = self.model.model.get_input_embeddings()(effector_translation_input_ids)
        
                    

                    effector_target_translation_input_ids = effector_target_translation[:, idx, :].clone()
                    effector_target_translation_embeddings = self.model.model.get_input_embeddings()(effector_target_translation_input_ids)
                    
                    image_embeddings = frames_embeddings[idx]
                    
                    image_label = torch.ones((b, image_embeddings.shape[1]), dtype=torch.long).to(image.device) * (-100)
                    effector_target_translation_label = torch.ones((b, effector_target_translation_embeddings.shape[1]), dtype=torch.long).to(image.device) * (-100)
                    effector_translation_label = torch.ones((b, effector_translation_embeddings.shape[1]), dtype=torch.long).to(image.device) * (-100)

                    total_embeddings = torch.cat((total_embeddings, image_embeddings, effector_translation_embeddings,
                                                effector_target_translation_embeddings, action_embeddings), dim = 1 )
                    
                    #prepend -100 to labels for image patches
                    labels = torch.cat((labels.long(), 
                                        image_label.long(),
                                        effector_translation_label.long(),
                                        effector_target_translation_label.long(),
                                            action_input_ids.clone().long()), dim=1)
                    # exit()
                    # Generate the next token
                    # TODO: replace image with image special token in label
                    
                output = self.model(
                    inputs_embeds = total_embeddings,
                    labels = labels,
                )
                loss1 = output.loss
                

         
            # return {"loss": total_loss, "pred": pred, "image": image, 'mask': mask, 'predict_los': loss}
        if calvin_samples:
            image = calvin_samples["video"].to(torch.float32)
            instructions = calvin_samples["instructions"].to(image.device)

            actions = calvin_samples["actions"].to(image.device)
            robot_state = calvin_samples["robot_state"].to(image.device)
            
            max_length = image.shape[1]
            # print('start training')
            # print(samples.keys())
        
        
            
            with autocast(dtype=torch.float16):
                instruction_embeds = self.model.model.get_input_embeddings()(instructions)
                # total_embeddings = torch.empty((instruction_embeds.shape[0], 0, self.model_size), dtype=torch.float16).to(image.device)
                total_embeddings = instruction_embeds.clone()
                
                frames_embeddings = []
                b, t, c, h, w = image.shape
                video_batch = image.reshape(b * t, c, h, w)

                # if epoch < 60 and self.training:
                features, mask, ids_restore =   self.visual_encoder(video_batch.unsqueeze(1), use_mask = True) # (N, 1, C, H, W)
                image_embeds = self.ln_vision(features)
                if not finetune:
                    pred = self.visual_encoder.forward_decoder(image_embeds, ids_restore)
                    mae_loss +=   self.visual_encoder.forward_loss(video_batch.unsqueeze(1), pred, mask)
                # else:
                #     features, mask, ids_restore =   self.visual_encoder(video_batch.unsqueeze(1), use_mask = False)
                #     image_embeds = self.ln_vision(features)
                #     pred = None
                
                image_embeds = image_embeds.reshape( b, t, image_embeds.shape[-2], image_embeds.shape[-1])
                image_embeds = self.linear_projection(image_embeds)

                for i in range(t):
                    frames_embeddings.append(image_embeds[:, i, :, :])
                

                # total_embeddings = torch.empty((b, 0, 768), dtype=torch.float16).to(image.device)
                
                
                labels = instructions.clone().to(image.device)
                for idx in range(max_length):  
                    action_input_ids = actions[:, idx, :].clone()         
                    action_embeddings = self.model.model.get_input_embeddings()(action_input_ids)
                    
                    robot_state_input_ids = robot_state[:, idx, :].clone()
                    robot_state_embeddings = self.model.model.get_input_embeddings()(robot_state_input_ids)
        
                    image_embeddings = frames_embeddings[idx]
                    
                    image_label = torch.ones((b, image_embeddings.shape[1]), dtype=torch.long).to(image.device) * (-100)
                    robot_state_embeddings_label = torch.ones((b, robot_state_embeddings.shape[1]), dtype=torch.long).to(image.device) * (-100)
                   

                    total_embeddings = torch.cat((total_embeddings, image_embeddings,
                                                robot_state_embeddings, action_embeddings), dim = 1 )
                    
                    #prepend -100 to labels for image patches
                    labels = torch.cat((labels.long(), 
                                        image_label.long(),
                                        robot_state_embeddings_label.long(),
                                            action_input_ids.clone().long()), dim=1)
                    # exit()
                    # Generate the next token
                    # TODO: replace image with image special token in label
                    
                output = self.model(
                    inputs_embeds = total_embeddings,
                    labels = labels,
                )
                loss2 = output.loss
        # processing minecraft data
        if minecraft_samples:
            if "image" in minecraft_samples.keys():
                image = minecraft_samples["image"]
            elif "video" in minecraft_samples.keys():
                image = minecraft_samples["video"]
            elif 'observations' in minecraft_samples.keys():
                image = minecraft_samples['observations']
            else:
                raise ValueError("No image or video input found in input dict.")
            
            # image = image.to(torch.float32)
            
            text_output = minecraft_samples["action"]
            instructions = minecraft_samples["instructions"]
            for i in range(len(text_output)):
                text_output[i] = text_output[i].replace('[startaction]', '[STARTACTION]')
                text_output[i] = text_output[i].replace('[endofaction]', '[ENDOFACTION]')
                text_output[i] = text_output[i].replace('camerax', 'CAMERAX')
                text_output[i] = text_output[i].replace('cameray', 'CAMERAY')
            
            instructions_tokens = self.tokenizer(
                        instructions,
                        padding='longest',
                        truncation=True,
                        max_length=self.num_frames * 100,
                        add_special_tokens = True,
                        # skip_special_tokens=False,
                        return_tensors="pt"
                    ).to(image.device)
            
            pattern = r'(.*?\[ENDOFACTION\])'
            text_output_final = [ [s for s in re.split(pattern, text) if s] for text in text_output ]

            num_frames = self.num_frames
            max_length = self.num_frames

            
            
            frames_embeddings = []
            b, t, c, h, w = image.shape
            video_batch = image.reshape(b * t, c, h, w)
            with autocast(dtype=torch.float16):
                instruction_embeds = self.model.model.get_input_embeddings()(instructions_tokens.input_ids)

                features, mask, ids_restore =   self.visual_encoder(video_batch.unsqueeze(1)) # (N, 1, C, H, W)
                image_embeds = self.ln_vision(features)

                if not finetune:
                    pred = self.visual_encoder.forward_decoder(image_embeds, ids_restore)
                    mae_loss +=   self.visual_encoder.forward_loss(video_batch.unsqueeze(1), pred, mask)

                image_embeds = image_embeds.reshape( b, self.num_frames, image_embeds.shape[-2], image_embeds.shape[-1])

                for i in range(num_frames):
                    frames_embeddings.append(image_embeds[:, i, :, :])
                

                # total_embeddings = torch.empty((b, 0, 768)).to(image.device)
                total_embeddings = instruction_embeds.clone()
                
                # labels = torch.empty((b, 0), dtype=torch.long).to(image.device)
                labels = instructions_tokens.input_ids.clone().to(image.device)
                for idx in range(max_length):  
                    
                    text_output_idx = [ text[idx] for text in text_output_final ]
                    
                    # print('text_output_idx: ', text_output_idx)
                    target_tokens = self.tokenizer(
                        text_output_idx,
                        padding='longest',
                        truncation=True,
                        max_length=self.num_frames * 100,
                        add_special_tokens = False,
                        # skip_special_tokens=False,
                        return_tensors="pt"
                    ).to(image.device)
                    
                    input_ids = target_tokens.input_ids
                    # print('input ids: ', input_ids)
                    # exit()
                
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
                    loss3 = output.loss
        
        with autocast(dtype=torch.float16):
            if finetune:
                total_loss = loss1 + loss2 + loss3 
            else:
                total_loss = 2 * (loss1 + loss2 + loss3) + mae_loss
            return {"loss": total_loss}


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

        mae_deocder = cfg.get("mae_decoder", True)

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
            mae_decoder = mae_deocder
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
@registry.register_model("robot_transformer")
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