import os
import cv2
import pandas as pd
import numpy as np
from collections import Counter
# from base.base_dataset import TextVideoDataset
from lavis.datasets.datasets.trio_video_caption_dataset import TrioVideoCaptionDataset
import json
import random
from torchvision import transforms
from lavis.datasets.datasets.base_dataset import BaseDataset
import tensorflow_datasets as tfds

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
    
def dataset2path(dataset_name):
  if dataset_name == 'robo_net':
    version = '1.0.0'
  elif dataset_name == 'language_table':
    version = '0.0.1'
  else:
    version = '0.1.0'
  return f'gs://gresearch/robotics/{dataset_name}/{version}'


def as_gif(images, path='temp.gif'):
  # Render the images as the gif:
  images[0].save(path, save_all=True, append_images=images[1:], duration=1000, loop=0)
  gif_bytes = open(path,'rb').read()
  return gif_bytes


class RobotDatasetAMLT(BaseDataset):
    """

    """
    def __init__(self,  total_num_frames=4, scale='small', split='train'):
        # use_fixed_start=True means we use the start time of the segment as the start time of the video
        # super().__init__(*args, **kwargs)

         
        self.total_num_frames = total_num_frames
        self.scale = scale

        DATASET_VERSION = '0.0.1'
        DATASET_NAME = 'language_table_sim'  # CHANGEME: change this to load another dataset.

        dataset_directories = {
            'language_table': 'gs://gresearch/robotics/language_table',
            'language_table_sim': 'gs://gresearch/robotics/language_table_sim',
            'language_table_blocktoblock_sim': 'gs://gresearch/robotics/language_table_blocktoblock_sim',
            'language_table_blocktoblock_4block_sim': 'gs://gresearch/robotics/language_table_blocktoblock_4block_sim',
            'language_table_blocktoblock_oracle_sim': 'gs://gresearch/robotics/language_table_blocktoblock_oracle_sim',
            'language_table_blocktoblockrelative_oracle_sim': 'gs://gresearch/robotics/language_table_blocktoblockrelative_oracle_sim',
            'language_table_blocktoabsolute_oracle_sim': 'gs://gresearch/robotics/language_table_blocktoabsolute_oracle_sim',
            'language_table_blocktorelative_oracle_sim': 'gs://gresearch/robotics/language_table_blocktorelative_oracle_sim',
            'language_table_separate_oracle_sim': 'gs://gresearch/robotics/language_table_separate_oracle_sim',
        }

        dataset_path = os.path.join(dataset_directories[DATASET_NAME], DATASET_VERSION)
        builder = tfds.builder_from_directory(dataset_path)
        self.episode_ds = builder.as_dataset(split=split)
        
        
    def __len__(self):
        return len(self.episode_ds)
    
    
    def __getitem__(self, index):


        video_path = ann["video"]
        #video_path = video_path[1:] # removing the '.' prefix
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