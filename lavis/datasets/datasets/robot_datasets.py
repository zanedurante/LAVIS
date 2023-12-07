import os
import cv2
import pandas as pd
import numpy as np
from collections import Counter
# from base.base_dataset import TextVideoDataset
import json
import random
from torchvision import transforms
# from lavis.datasets.datasets.base_dataset import BaseDataset
import tensorflow_datasets as tfds
# import mediapy
import tensorflow as tf
import base64
import matplotlib.pyplot as plt
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


class RobotDatasetAMLT():
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

        # self.steps = self.episode_ds.flat_map(lambda x: x['steps'])
        return self.episode_ds.skip(index).take(1)
# /home/nikepupu/Desktop/amlt/LAVIS/lavis/datasets/datasets/robot_datasets.py

def decode_inst(inst):
  """Utility to decode encoded language instruction"""
  return bytes(inst[np.where(inst != 0)].tolist()).decode("utf-8") 

if __name__ == "__main__":
    # ds = RobotDatasetAMLT()
    # print(len(ds))
    # episodes = next(iter(ds[0]))
    # frames = []
    # for step in episodes['steps'].as_numpy_iterator():
    #     frames.append(step['observation']['rgb'])
    # print('frames', len(frames))
    # print('frames: ', frames)
    # mediapy.show_video(frames, fps=5)

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
    import torch
    from tqdm import tqdm
    dataset_path = os.path.join(dataset_directories[DATASET_NAME], DATASET_VERSION)
    # load raw dataset --> replace this with tfds.load() on your
    # local machine!
    b = tfds.builder_from_directory(dataset_path)
    ds = b.as_dataset(split='train[:1000]')
    total = len(ds)
    ds = tfds.as_numpy(ds)

    def episode2steps(episode):
        return episode['steps']


    # convert RLDS episode dataset to individual steps & reformat
    # ds = ds.map(
    #     episode2steps, num_parallel_calls=tf.data.AUTOTUNE).flat_map(lambda x: x)
    # ds = ds.map(step_map_fn, num_parallel_calls=tf.data.AUTOTUNE)

    # shuffle, repeat, pre-fetch, batch
    # ds = ds.cache()         # optionally keep full dataset in memory
    # ds = ds.shuffle(100)    # set shuffle buffer size
    # ds = ds.repeat()        # ensure that data never runs out
    base_path = os.path.expanduser('~/dataset/language_table1kfull/')
    
    os.makedirs(base_path, exist_ok=True)
    l_m = 0
    l_dist = []
    eet = []
    x_max = float('-inf')
    x_min = float('inf')

    y_max = float('-inf')
    y_min = float('inf')
    for idx, batch in tqdm(enumerate(ds), total=total):
        # here you would add your Jax / PyTorch training code
        # if i == 10000: break
        steps = batch['steps']
        episode_id = batch['episode_id']
        decoded_bytes = episode_id.decode('utf-8')
        episode_id = decoded_bytes
        # check if ~/dataset/language_table/ exist if not exist create it
        
        trajectory = []
        obs = []
        l = len(steps)
        l_m = max(l, l_m)
        l_dist.append(l)

        chunk_id = 0
        chunk_size = 9
        
        for step in steps:
            
            t = {
                'observation': step['observation']['rgb'],
                'action': step['action'],
                'effector_target_translation': step['observation']['effector_target_translation'],
                'effector_translation': step['observation']['effector_translation'],
                'instruction': step['observation']['instruction'],
                'is_first': step['is_first'],
                'is_last': step['is_last'],
                'is_terminal': step['is_terminal'],
            }
            instruction = step['observation']['instruction']
            instruction = ''.join(chr(id) for id in instruction if id != 0)
            x_max = max(x_max, step['observation']['effector_translation'][0])
            x_min = min(x_min, step['observation']['effector_translation'][0])

            y_max = max(y_max, step['observation']['effector_translation'][1])
            y_min = min(y_min, step['observation']['effector_translation'][1])

            eet.append(step['observation']['effector_translation'][0])
        #     print(instruction, 'is last: ', step['is_last'], 'is_first: ', step['is_first'], 'is_terminal: ', step['is_terminal'])
        # print('====')
        # obs.append(step['observation']['rgb'])
           
            trajectory.append(t)
            if len(trajectory) == chunk_size:
                np.savez_compressed(os.path.join(base_path, f'{episode_id}_{chunk_id}.npz'), trajectory=trajectory)
                trajectory = []
                chunk_id += 1


    #     # # save the episode id and the step id to local as npz file
        if len(trajectory) > 0:
            np.savez_compressed(os.path.join(base_path, f'{episode_id}_{chunk_id}.npz'), trajectory=trajectory)
    # print(l_m)
    plt.hist(eet, bins=100)
    plt.show()
print(x_max)
print(x_min)

print(y_min)
print(y_max)
