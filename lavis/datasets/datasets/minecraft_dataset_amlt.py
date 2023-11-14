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

KEYBOARD_BUTTON_MAPPING = {
    "key.keyboard.escape" :"ESC",
    "key.keyboard.s" :"back",
    "key.keyboard.q" :"drop",
    "key.keyboard.w" :"forward",
    "key.keyboard.1" :"hotbar.1",
    "key.keyboard.2" :"hotbar.2",
    "key.keyboard.3" :"hotbar.3",
    "key.keyboard.4" :"hotbar.4",
    "key.keyboard.5" :"hotbar.5",
    "key.keyboard.6" :"hotbar.6",
    "key.keyboard.7" :"hotbar.7",
    "key.keyboard.8" :"hotbar.8",
    "key.keyboard.9" :"hotbar.9",
    "key.keyboard.e" :"inventory",
    "key.keyboard.space" :"jump",
    "key.keyboard.a" :"left",
    "key.keyboard.d" :"right",
    "key.keyboard.left.shift" :"sneak",
    "key.keyboard.left.control" :"sprint",
    "key.keyboard.f" :"swapHands",
}

# Template action
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
    "camera": np.array([0, 0]),
}

MESSAGE = """
This script will take a video, predict actions for its frames and
and show them with a cv2 window.

Press any button the window to proceed to the next frame.
"""

# Matches a number in the MineRL Java code regarding sensitivity
# This is for mapping from recorded sensitivity to the one used in the model
CAMERA_SCALER = 360.0 / 2400.0


def top_k_common_items(lst, k):
    count = Counter(lst)
    return [item for item, _ in count.most_common(k)]

def json_action_to_env_action(json_action):
    """
    Converts a json action into a MineRL action.
    Returns (minerl_action, is_null_action)
    """
    # This might be slow...
    env_action = NOOP_ACTION.copy()
    # As a safeguard, make camera action again so we do not override anything
    env_action["camera"] = np.array([0, 0])

    is_null_action = True
    keyboard_keys = json_action["keyboard"]["keys"]
    for key in keyboard_keys:
        # You can have keys that we do not use, so just skip them
        # NOTE in original training code, ESC was removed and replaced with
        #      "inventory" action if GUI was open.
        #      Not doing it here, as BASALT uses ESC to quit the game.
        if key in KEYBOARD_BUTTON_MAPPING:
            env_action[KEYBOARD_BUTTON_MAPPING[key]] = 1
            is_null_action = False

    mouse = json_action["mouse"]
    camera_action = env_action["camera"]
    camera_action[0] = mouse["dy"] * CAMERA_SCALER
    camera_action[1] = mouse["dx"] * CAMERA_SCALER

    if mouse["dx"] != 0 or mouse["dy"] != 0:
        is_null_action = False
    else:
        if abs(camera_action[0]) > 180:
            camera_action[0] = 0
        if abs(camera_action[1]) > 180:
            camera_action[1] = 0

    mouse_buttons = mouse["buttons"]
    if 0 in mouse_buttons:
        env_action["attack"] = 1
        is_null_action = False
    if 1 in mouse_buttons:
        env_action["use"] = 1
        is_null_action = False
    if 2 in mouse_buttons:
        env_action["pickItem"] = 1
        is_null_action = False

    return env_action, is_null_action

def get_video_duration(video_path):
    cap = cv2.VideoCapture(video_path)
    
    # Check if video loaded successfully
    if not cap.isOpened(): 
        print("Error: Could not read video file")
        return None
    
    # Get frame count
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Get frames per second (fps)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Calculate duration
    duration_s = frame_count / fps if fps > 0 else 0
    
    # Release the video capture object
    cap.release()
    
    return duration_s, fps, frame_count

class MinecraftVidDatasetAMLT(TrioVideoCaptionDataset):
    """
    MinecraftVid Dataset.
    Assumes MinecraftVid data is structured as follows.
    minecraft/
           
        1.mp4           (videoid.mp4)
        1.jsonl
        ...

    """
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths, num_skip_frames=None, total_num_frames=4, scale='small'):
        # use_fixed_start=True means we use the start time of the segment as the start time of the video
        # super().__init__(*args, **kwargs)
         
         self.total_num_frames = total_num_frames
         self.transforms = self.get_transforms()
         self.scale = scale
         super().__init__(vis_processor, text_processor, vis_root, ann_paths, num_skip_frames, total_num_frames)
        

    def _load_metadata(self):
        assert self.scale in ['small', 'medium', 'large', 'tiny']
        # self.converted_csv_path = f"/mnt/datasets_mnt/metadata_{self.scale}.csv"
        self.converted_csv_path = f"/mnt/datasets_mnt/metadata_9.csv"
        self.metadata = pd.read_csv(self.converted_csv_path)
        # print(self.metadata)
        # exit()
        
    def __len__(self):
        return len(self.metadata)
    
    
    def __getitem__(self, index):

        ann = self.metadata.iloc[index]

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