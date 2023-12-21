
# from lavis.datasets.datasets.base_dataset import BaseDataset
import os
import numpy as np
import torch
import torch.nn.functional as F
from lavis.datasets.datasets.base_dataset import BaseDataset
from transformers import AutoTokenizer
from torchvision import transforms
import pandas as pd
import cv2
from collections import Counter
from lavis.datasets.datasets.trio_video_caption_dataset import TrioVideoCaptionDataset
from torch.utils.data.dataloader import default_collate
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
            # transforms.RandomResizedCrop(input_res, scale=randcrop_scale),
            transforms.Resize((input_res, input_res)),
            transforms.ColorJitter(brightness=color_jitter[0], saturation=color_jitter[1], hue=color_jitter[2]),
            normalize,
        ]),
        'val': transforms.Compose([
            # transforms.Resize(center_crop),
            # transforms.CenterCrop(center_crop),
            # transforms.Resize(input_res),
            transforms.Resize((input_res, input_res)),
            normalize,
        ]),
        'test': transforms.Compose([
            # transforms.Resize(center_crop),
            # transforms.CenterCrop(center_crop),
            # transforms.Resize(input_res),
            transforms.Resize((input_res, input_res)),
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

class PretrainingDatasetAMLT(TrioVideoCaptionDataset):
    """

    """
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths, num_skip_frames=None, total_num_frames=4):
        self.basedir = '/mnt/languagetablesim'
        # load all npz files under the directory
        # self.episodes = []
        print('loading metadata for robot')
        with open(os.path.join(self.basedir, 'robot_small.txt'), 'r') as f:
            self.files = f.read().splitlines() 
        print('loading metadata for robot done')
        self.files = sorted(self.files)
        total = len(self.files)
        self.files = self.files
                
        self.base_model_name = "facebook/opt-125m"
        self.tokenizer =  AutoTokenizer.from_pretrained(self.base_model_name)
        for i in range(21):
            self.tokenizer.add_tokens([f"[ROBOTACTIONX{i}]", f"[ROBOTACTIONY{i}]"])
            self.tokenizer.add_tokens([f"[ROBOTEETX{i}]", f"[ROBOTEETY{i}]"])
            self.tokenizer.add_tokens([f"[ROBOTEETTX{i}]", f"[ROBOTEETTY{i}]"])
        
        self.tokenizer.add_tokens(['[ENDOFACTION]'])
        self.tokenizer.add_tokens(['[STARTACTION]'])
        self.tokenizer.add_tokens(['[TERMINAL]'])
        
        self.tokenizer.add_tokens(['[STARTEET]'])
        self.tokenizer.add_tokens(['[ENDOFEET]'])

        self.tokenizer.add_tokens(['[STARTEETT]'])
        self.tokenizer.add_tokens(['[ENDOFEETT]'])

        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.transforms = self.get_transforms()
        # ================== load metadata ==================
        print('loading metadata for minecraft')
        self.converted_csv_path = f"/mnt/datasets_mnt/metadata_9_small.csv"
        self.metadata = pd.read_csv(self.converted_csv_path)
        print('loading metadata for minecraft done')
        self.total_num_frames = total_num_frames
        super().__init__(vis_processor, text_processor, vis_root, ann_paths, num_skip_frames, total_num_frames)

    def _load_metadata(self):
        pass

    def __len__(self):
        return len(self.files) + len(self.metadata)
    
    def collater(self, samples):
        # print('start collate')
        minecraft_samples = []
        robot_samples = []
        # print('samples in collate:  ', samples)
        for sample in samples:
            # print(sample.keys())
            if sample['type'] == 'minecraft':
                del sample['effector_target_translation']
                del sample['effector_translation']
                minecraft_samples.append(sample)
            else:
                robot_samples.append(sample)

        if len(minecraft_samples) > 0:
            minecraft_return = default_collate(minecraft_samples)
        else:
            minecraft_return = None

        if len(robot_samples) > 0:
            obs_batch = []
            instrs_batch = []
            actions_batch = []
            effector_target_translation = []
            effector_translation = []
            m_obs = 9

            # Function to pad a tensor to a target size
            def pad_tensor(input_tensor, target_size):
                
                # Calculate padding size
                pad_size = target_size - input_tensor.size(1)
                last_observation = input_tensor[:, -1, :, :, :].unsqueeze(1).repeat(1, pad_size, 1, 1, 1)
                # Apply padding
                padded_tensor = torch.cat((input_tensor, last_observation), dim=1)
                return padded_tensor
            
                # return F.pad(input_tensor, (0, 0, 0, 0, 0, 0, 0, pad_size), "constant", 0)

            for sample in robot_samples:     
                # print(sample)       
                obs =  pad_tensor(torch.tensor(np.array(sample['observations'])).unsqueeze(0), m_obs)
                obs_batch.append(obs)
                instr = sample['instructions'][0]
                instr = ''.join(chr(id) for id in instr if id != 0)
                instrs_batch.append(instr)
                actions = sample['action']
                effector_target_translation.append(sample['effector_target_translation'])
                effector_translation.append(sample['effector_translation'])
                # print('actions: ', actions)
                
                actions_batch.append(actions)

            
            for batch in actions_batch:
                while len(batch) < m_obs:
                    batch.append('[STARTACTION][TERMINAL][TERMINAL][ENDOFACTION]')
            
            for batch in effector_target_translation:
                while len(batch) < m_obs:
                    batch.append(batch[-1])
            
            for batch in effector_translation:
                while len(batch) < m_obs:
                    batch.append(batch[-1])

            
            obs_batch = torch.cat(obs_batch, dim=0)
            obs_batch = obs_batch.permute(0, 1, 4, 2, 3)
            
            instrs_batch = self.tokenizer(instrs_batch, padding='longest', return_tensors="pt", 
                                        truncation=True, max_length=200,  add_special_tokens = False)
            instrs_batch = instrs_batch.input_ids
            max_len = max([len(action) for action in actions_batch])
            for action_lst in actions_batch:
                action_lst.extend(['[ENDOFACTION]'] * (max_len - len(action_lst)))



            a_batch = []
            for action_batch in actions_batch:
                tmp = self.tokenizer(action_batch, padding='longest', return_tensors="pt", 
                                    truncation=True, max_length=200, add_special_tokens = False)
                tmp = tmp.input_ids
                a_batch.append(tmp)

            eet_batch = []
            for action_batch in effector_target_translation:
                tmp = self.tokenizer(action_batch, padding='longest', return_tensors="pt", 
                                    truncation=True, max_length=200, add_special_tokens = False)
                tmp = tmp.input_ids
                eet_batch.append(tmp)
            
            et_batch = []
            for action_batch in effector_translation:
                tmp = self.tokenizer(action_batch, padding='longest', return_tensors="pt", 
                                    truncation=True, max_length=200, add_special_tokens = False)
                tmp = tmp.input_ids
                et_batch.append(tmp)
            # print('a batch: ')
            # print(a_batch)
            a_batch = torch.stack(a_batch, dim=0) 
            obs_batch = obs_batch.float() / 255

            obs_batch = torch.stack([
                torch.stack([
                    self.transforms(frame) for frame in video_sequence
                ]) for video_sequence in obs_batch
            ])

            eet_batch = torch.stack(eet_batch, dim=0) 
            et_batch = torch.stack(et_batch, dim=0)

            # print(a_batch.shape) 
            # print(obs_batch.shape)
            # print(instrs_batch)
            # exit()
            robot_return =  {
                'video':  obs_batch, 
                'instructions': instrs_batch, 
                'actions': a_batch,
                'effector_target_translation': eet_batch,
                'effector_translation': et_batch
            }
        else:
            robot_return = None

        # print(minecraft_return)
        # print(robot_return)
        return {
            'minecraft': minecraft_return,
            'robot': robot_return
        }
        
    
    def get_bin_id(self, number, range_min, range_max, num_bins):
        """
        Function to find the bin ID for a given number in a specified range divided into bins.

        :param number: The number for which the bin ID is to be found.
        :param range_min: The minimum value of the range.
        :param range_max: The maximum value of the range.
        :param num_bins: The total number of bins in the range.
        :return: The bin ID in which the given number falls.
        """
        # Check if the number is within the range
        if number < range_min or number > range_max:
           # clip into the range
            number = min(max(number, range_min), range_max)

        # Calculate the width of each bin
        bin_width = (range_max - range_min) / num_bins

        # Calculate the bin ID
        bin_id = int((number - range_min) / bin_width)

        return bin_id
    
    def get_transforms(self):
        return get_transforms("train")
    

    def __getitem__(self, index):
        import random
        dataset_type = random.choice(['minecraft', 'robot'])

        if dataset_type == 'minecraft':
            index_to_use = random.randint(0, len(self.metadata)-1)
            return self.getitem1(index_to_use)
        else:
            index_to_use = random.randint(0, len(self.files)-1)
            return self.getitem2(index_to_use)
    

    def getitem1(self, index):

        while True:
            try:
                ann = self.metadata.iloc[index]

                video_path = ann["video"]
                # video_path = video_path[1:] # removing the '.' prefix
                start_frame = ann.get("start_frame", 0)
                end_frame = ann.get("end_frame", -1)


                video = self._load_video(video_path, start_frame, end_frame)
                video = self.transforms(video)
                caption = self.text_processor(ann["caption"])

                input_text = self._get_next_prompt() # Inherited from CaptionDataset
                break
            except:
                index = (index+1) % len(self.metadata)
        # print('video: ', video.shape)
        # "image_id" is kept to stay compatible with the COCO evaluation format
        return {
            "observations": video,
            "instructions": input_text, # Input prompt
            "action": caption, # Correct caption
            "type": 'minecraft',
            "effector_target_translation": None,
            "effector_translation": None
        }
    
    
    def getitem2(self, index):
        file = self.files[index]
        episode = np.load(os.path.join(self.basedir, file), allow_pickle=True)
        trajectory = episode['trajectory']

        obs = []
        instrs = []
        actions = []
        effector_translations = []
        effector_target_translation = [] 
        types = []
        
        for step in trajectory:
            
            obs.append(step['observation'])
            instruction = step['instruction'] 
            instruction = ''.join(chr(id) for id in instruction)
            instrs.append(step['instruction'])
            ee_t_first_dim =  int(self.get_bin_id(step['effector_translation'][0], 0.15, 0.6, 20))
            ee_t_second_dim = int(self.get_bin_id(step['effector_translation'][1], -0.3, 0.3, 20))
            effector_translations.append(f"[STARTEET][ROBOTEETX{ee_t_first_dim}][ROBOTEETY{ee_t_second_dim}][ENDOFEET]")

            ee_tt_first_dim =  int(self.get_bin_id(step['effector_target_translation'][0], 0.15, 0.6, 20))
            ee_tt_second_dim = int(self.get_bin_id(step['effector_target_translation'][1], -0.3, 0.3, 20))
            effector_target_translation.append(f"[STARTEETT][ROBOTEETTX{ee_tt_first_dim}][ROBOTEETTY{ee_tt_second_dim}][ENDOFEETT]")

            if not step['is_terminal']:
                first_dim =  int(self.get_bin_id(step['action'][0], -0.03, 0.03, 20))
                second_dim = int(self.get_bin_id(step['action'][1], -0.03, 0.03, 20))
                actions.append(f"[STARTACTION][ROBOTACTIONX{first_dim}][ROBOTACTIONY{second_dim}][ENDOFACTION]")
            else:
                actions.append('[STARTACTION][TERMINAL][TERMINAL][ENDOFACTION]')
            types.append('robot')

        # return obs, instrs, actions, is_firsts, is_lasts, is_terminals

        to_return =  {
            "observations": obs,
            "instructions": instrs, 
            "action": actions,
            "effector_translation": effector_translations,
            "effector_target_translation": effector_target_translation,
            "type": 'robot',
        }

        return to_return