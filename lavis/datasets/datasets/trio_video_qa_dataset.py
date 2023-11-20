"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import os
from lavis.datasets.datasets.base_dataset import BaseDataset

from lavis.datasets.datasets.caption_datasets import CaptionDataset
from lavis.datasets.datasets.video_caption_datasets import VideoCaptionDataset, VideoCaptionEvalDataset
from lavis.datasets.datasets.trio_video_caption_dataset import TrioVideoCaptionDataset, TrioVideoCaptionEvalDataset
import decord
import numpy as np
from abc import abstractmethod
from torchvision import transforms
from PIL import Image


decord.bridge.set_bridge('torch')


def init_transform_dict(input_res=224,
                        center_crop=256,
                        randcrop_scale=(0.5, 1.0),
                        color_jitter=(0, 0, 0),
                        norm_mean=(0.485, 0.456, 0.406),
                        norm_std=(0.229, 0.224, 0.225),
                        use_clip_norm=True):
    # Use normalization from: https://github.com/openai/CLIP/blob/a1d071733d7111c9c014f024669f959182114e33/clip/clip.py#L83
    if use_clip_norm:
        norm_mean = (0.48145466, 0.4578275, 0.40821073)
        norm_std = (0.26862954, 0.26130258, 0.27577711)
    normalize = transforms.Normalize(mean=norm_mean, std=norm_std)
    tsfm_dict = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(input_res, scale=randcrop_scale),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=color_jitter[0], saturation=color_jitter[1], hue=color_jitter[2]),
            normalize,
        ]),
        'val': transforms.Compose([
            transforms.Resize(center_crop),
            transforms.CenterCrop(center_crop),
            transforms.Resize(input_res),
            normalize,
        ]),
        'test': transforms.Compose([
            transforms.Resize(center_crop),
            transforms.CenterCrop(center_crop),
            transforms.Resize(input_res),
            normalize,
        ])
    }
    return tsfm_dict


def get_transforms(split):
    if split in ['train', 'val', 'test']:
        return init_transform_dict()[split]
    else:
        raise ValueError('Split {} not supported.'.format(split))


class TrioVideoQADataset(TrioVideoCaptionDataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths, num_skip_frames=-1, total_num_frames=4, prompt_type="image"):
        """
        TrioVideoCaptionDataset structure supports fixed FPS and random frame sampling during training through an interface.
        Use num_skip_frames to set the number of frames to skip for fixed FPS sampling. If -1, assumes random frame sampling. If 0, uses every frame, if 1 uses every other frame, etc.
        total_num_frames is the total number of frames to sample from a video for each clip during training. This is equivalent to the number of frames in a video during inference.
        For _load_annotations, you need to either load a CSV file or create a pd.DataFrame with the following structure:
            video_path, question, answer, start_frame (optional), end_frame (optional)
        split (string): val or test
        """
        super().__init__(vis_processor, text_processor, vis_root, ann_paths, num_skip_frames, total_num_frames, prompt_type)
        self.prompts = [
            "Question: {} Short answer: "
        ]

        self.prompt_idx = 0 # Index of the current prompt

    def _get_next_prompt(self, index):
        self.prompt_idx += 1
        self.prompt_idx = self.prompt_idx % len(self.prompts)
        inp_prompt = self.prompts[self.prompt_idx]
        inp_prompt = inp_prompt.format(self.metadata.iloc[index]["question"])
        return inp_prompt


    def __getitem__(self, index):

        ann = self.metadata.iloc[index]
        video_path = ann["video_path"]
        start_frame = ann.get("start_frame", 0)
        end_frame = ann.get("end_frame", -1)

        video = self._load_video(video_path, start_frame, end_frame)
        video = self.transforms(video)
        caption = self.text_processor(ann["answer"])

        input_text = self._get_next_prompt(index) # Inherited from CaptionDataset

        # "image_id" is kept to stay compatible with the COCO evaluation format
        return {
            "video": video,
            "text_input": input_text, # Input prompt
            "text_output": caption, # Correct caption
        }


class TrioVideoQAEvalDataset(TrioVideoCaptionEvalDataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths, num_skip_frames=-1, total_num_frames=4, prompt_type="image"):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        split (string): val or test
        """
        super().__init__(vis_processor, text_processor, vis_root, ann_paths, num_skip_frames, total_num_frames) # Note, we keep vis_processor here for compatibility with the original code
        self.prompts = [
            "Question: {} Short answer: "
        ]

    def _get_next_prompt(self, index):
        self.prompt_idx += 1
        self.prompt_idx = self.prompt_idx % len(self.prompts)
        inp_prompt = self.prompts[self.prompt_idx]
        inp_prompt = inp_prompt.format(self.metadata.iloc[index]["question"])
        return inp_prompt

    def __getitem__(self, index):

        ann = self.metadata.iloc[index]

        video_path = ann["video_path"]
        start_frame = ann.get("start_frame", 0)
        end_frame = ann.get("end_frame", -1)

        video = self._load_video(video_path, start_frame, end_frame)
        video = self.transforms(video)
        caption = self.text_processor(ann["answer"])

        input_text = self._get_next_prompt(index) # Inherited from CaptionDataset

        # "image_id" is kept to stay compatible with the COCO evaluation format
        return {
            "video": video,
            "text_input": input_text, # Input prompt
            "text_output": caption, # Correct caption
        }