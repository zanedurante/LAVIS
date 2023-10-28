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


class TrioVideoCaptionDataset(VideoCaptionDataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths, num_skip_frames=-1, total_num_frames=4):
        """
        TrioVideoCaptionDataset structure supports fixed FPS and random frame sampling during training through an interface.
        Use num_skip_frames to set the number of frames to skip for fixed FPS sampling. If -1, assumes random frame sampling. If 0, uses every frame, if 1 uses every other frame, etc.
        total_num_frames is the total number of frames to sample from a video for each clip during training. This is equivalent to the number of frames in a video during inference.
        For _load_annotations, you need to either load a CSV file or create a pd.DataFrame with the following structure:
            video, caption, start_frame (optional), end_frame (optional)
        split (string): val or test
        """
        self.text_processor = text_processor
        self.vis_processor = vis_processor
        self.total_num_frames = total_num_frames
        self.num_skip_frames = num_skip_frames
        self._load_metadata()
        self.annotation = self.metadata # For compatibility with the original code
        self.transforms = self.get_transforms()
        self.prompts = [
            # "A short image caption: ",
            # "A short image description: ",
            # "A photo of ",
            # "An image that shows ",
            # "Write a short description of the image. ",
            # "Write a description for the photo.",
            # "Provide a description of what is presented in the photo. ",
            # "Briefly describe the content of the image. ",
            # "Can you briefly explain what you see in the image? ",
            # "Could you use a few words to describe what you perceive in the photo? ",
            # "Please provide a short depiction of the picture. ",
            # "Using language, provide a short account of the image. ",
            # "Use a few words to illustrate what is happening in the picture. ",
            'A Minecraft video with action sequences of ',
            'The player play Minecraft with  the following actions: ',
            'A video showcasing action sequences in Minecraft: ',
            "The player engages in these actions while playing Minecraft: "

        ]
        self.prompt_idx = 0 # Index of the current prompt

    @abstractmethod
    def _load_metadata(self):
        """
        Load metadata from a CSV file or generate pd.DataFrame.  Resulting pandas dataframe should have structure:
            video, caption, start_frame (optional), end_frame (optional)
        """
        self.metadata = None
        pass

    def __len__(self):
        return len(self.metadata)

    def _load_video(self, video_path, start_frame=0, end_frame=-1):
        frame_indices = None
        try:
            video_reader = decord.VideoReader(video_path, num_threads=1)
        except:
            print("Error loading video: {}".format(video_path))
            imgs = Image.new('RGB', (224, 224), (0, 0, 0))
            imgs = transforms.ToTensor()(imgs).unsqueeze(0)
            # Repeat self.total_num_frames times in first dim
            imgs = imgs.repeat(self.total_num_frames, 1, 1, 1)
            return imgs
        video_length = len(video_reader)

        if self.num_skip_frames < 0: # Use random frame sampling
            frame_indices = np.random.randint(0, video_length, self.total_num_frames)
            frame_indices = np.sort(frame_indices)
        else:
            # TODO: Implement variable/random offsets for fixed FPS sampling
            frame_indices = np.arange(start_frame, end_frame, self.num_skip_frames + 1)[:self.total_num_frames]

        frames = video_reader.get_batch(frame_indices)
        frames = frames.float() / 255
        frames = frames.permute(0, 3, 1, 2)
        return frames
            
    def get_transforms(self):
        return get_transforms("train")

    def __getitem__(self, index):

        ann = self.metadata.iloc[index]

        video_path = ann["video"]
        start_frame = ann.get("start_frame", 0)
        end_frame = ann.get("end_frame", -1)

        video = self._load_video(video_path, start_frame, end_frame)
        video = self.transforms(video)
        caption = self.text_processor(ann["caption"])

        input_text = self._get_next_prompt() # Inherited from CaptionDataset
        # print('video: ', video.shape)
        # "image_id" is kept to stay compatible with the COCO evaluation format
        return {
            "video": video,
            "text_input": input_text, # Input prompt
            "text_output": caption, # Correct caption
        }


class TrioVideoCaptionEvalDataset(TrioVideoCaptionDataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths, num_skip_frames=-1, total_num_frames=4):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        split (string): val or test
        """
        super().__init__(vis_processor, text_processor, vis_root, ann_paths, num_skip_frames, total_num_frames) # Note, we keep vis_processor here for compatibility with the original code
        self.prompts = ["A photo of "] # Use single prompt for evals


    # Different data loading for evaluation
    def _load_video(self, video_path, start_frame=0, end_frame=-1):
        frame_indices = None
        video_reader = decord.VideoReader(video_path, num_threads=1)
        video_length = len(video_reader)

        if self.num_skip_frames < 0: # Use evenly spread frame sampling (for eval datasets)
            frame_indices = np.linspace(0, video_length, self.num_frames)
        else:
            # TODO: Implement variable/random offsets for fixed FPS sampling
            frame_indices = np.arange(start_frame, end_frame, self.num_skip_frames+1)[:self.total_num_frames]

        frames = video_reader.get_batch(frame_indices)
        frames = frames.float() / 255
        frames = frames.permute(0, 3, 1, 2)
        return frames

    def get_transforms(self):
        return get_transforms("test")