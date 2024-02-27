
# from lavis.datasets.datasets.base_dataset import BaseDataset
import os
import numpy as np
import torch
import torch.nn.functional as F
from lavis.datasets.datasets.base_dataset import BaseDataset
from transformers import AutoTokenizer
from torchvision import transforms
from lavis.models.model_utils import init_tokenizer

def init_transform_dict(input_res=224,
                        center_crop=256,
                        randcrop_scale=(0.5, 1.0),
                        color_jitter=((0.7, 1.4), (0.7, 1.4), (-0.05, 0.05)),
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
    
# Function to pad a tensor to a target size
def pad_tensor(input_tensor, target_size):
        
        
        # Calculate padding size
        pad_size = target_size - input_tensor.size(1)
        # print('pad_size: ', pad_size, input_tensor.size() )
        last_observation = input_tensor[:, -1, :, :, :].unsqueeze(1).repeat(1, pad_size, 1, 1, 1)
        # Apply padding
        padded_tensor = torch.cat((input_tensor, last_observation), dim=1)
        return padded_tensor

import yaml
class CalvinDatasetAMLTTrain(BaseDataset):
    """

    """
    def __init__(self, finetune=True):
        self.calvin_basedir = '/mnt/calvin/training'
        # self.calvin_basedir = '/home/nikepupu/Desktop/calvin/dataset/calvin_debug_dataset/training'
        self.annotation = np.load(os.path.join(self.calvin_basedir, 'lang_annotations', 'auto_lang_ann.npy'), allow_pickle=True).item()
        with open(os.path.join(self.calvin_basedir,  'statistics.yaml'), 'r') as file:
            self.statistics = yaml.load(file, Loader=yaml.FullLoader)
        
        self.robot_obs_mean = self.statistics['robot_obs'][0]['mean']
        self.robot_obs_std = self.statistics['robot_obs'][0]['std']
        self.action_min = self.statistics['act_min_bound']
        self.action_max = self.statistics['act_max_bound']

        self.language =  self.annotation['language']['ann']
       
        self.annotated_episodis = self.annotation['info']['indx']

        def split_into_segments(ranges, languages):
            split_ranges = []
            for (start, end), lan in zip(ranges, languages):
                while start <= end:
                    # Calculate new end ensuring the range is at most 9 units
                    new_end = min(start + 3, end)
                    # Append the new segment
                    split_ranges.append((start, new_end, lan))
                    # Update start for next segment
                    start = new_end + 1
            return split_ranges
        
        self.annotated_episodis = split_into_segments(self.annotated_episodis, self.language)
        total = len(self.annotated_episodis)
        # self.annotated_episodis = self.annotated_episodis[:int(total * 0.9)]

        self.calvin_bin_size = 20
        
        bin_sizes = {
            'language_table': 20,
            'calvin': self.calvin_bin_size
        }
        self.base_model_name = "facebook/opt-125m"
        self.tokenizer = init_tokenizer(self.base_model_name, bin_sizes=bin_sizes)

        self.finetune=finetune
        self.transforms = self.get_transforms()
        
    def __len__(self):
        return len(self.annotated_episodis)
    
    def prepare_calvin_samples(self, calvin_samples):
        obs_batch = []
        instrs_batch = []
        actions_batch = []
        robot_states = []
        m_obs = 4
        
        for sample in calvin_samples:            
            obs =  pad_tensor(torch.tensor(np.array(sample['observations'])).unsqueeze(0), m_obs)
            obs_batch.append(obs)
            instr = sample['instructions'][0]
            instrs_batch.append(instr)
            actions = sample['action']
            robot_states.append(sample['state_obs'])
            
            actions_batch.append(actions)


        for batch in actions_batch:
            while len(batch) < m_obs:
                batch.append('[STARTACTION][TERMINAL][TERMINAL][ENDOFACTION]')
        
        for batch in robot_states:
            while len(batch) < m_obs:
                batch.append(batch[-1])
        
  
        
        obs_batch = torch.cat(obs_batch, dim=0)
        obs_batch = obs_batch.permute(0, 1, 4, 2, 3)
        
        instrs_batch = self.tokenizer(instrs_batch, padding='longest', return_tensors="pt", 
                                      truncation=True, max_length=200, add_special_tokens = False)
        instrs_batch = instrs_batch.input_ids
        max_len = max([len(action) for action in actions_batch])
        for action_lst in actions_batch:
            action_lst.extend(['[ENDOFACTION]'] * (max_len - len(action_lst)))



        a_batch = []
        for action_batch in actions_batch:
            tmp = self.tokenizer(action_batch, padding='longest', return_tensors="pt", truncation=True, max_length=200,  add_special_tokens = False)
            tmp = tmp.input_ids
            a_batch.append(tmp)

        state_batch = []
       
        for action_batch in robot_states:
            tmp = self.tokenizer(action_batch, padding='longest', return_tensors="pt", truncation=True, max_length=200,  add_special_tokens = False)
            tmp = tmp.input_ids
            state_batch.append(tmp)
        

        a_batch = torch.stack(a_batch, dim=0) 
        # print(obs_batch)
        # exit()
        obs_batch = obs_batch.float() / 255



        obs_batch = torch.stack([
            torch.stack([
                self.transforms(frame) for frame in video_sequence
            ]) for video_sequence in obs_batch
        ])

        state_batch = torch.stack(state_batch, dim=0) 
       
        return {
            'video':  obs_batch, 
            'instructions': instrs_batch, 
            'actions': a_batch,
            'robot_state': state_batch,
        }

    
    def collater(self, samples):
        # print('start collate')

        calvin_samples = []
        # print('samples in collate:  ', samples)
        for sample in samples:
            # print(sample.keys())
            del sample['effector_target_translation']
            del sample['effector_translation']
            calvin_samples.append(sample)
         
        
        if len(calvin_samples) > 0:
            
            calvin_return = self.prepare_calvin_samples(calvin_samples)
        else:
            calvin_return = None

        # print(minecraft_return)
        # print(robot_return)
        return {
            'calvin': calvin_return,
            'finetune': self.finetune
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
        start, end, language = self.annotated_episodis[index]
        # format start index to something like this: episode_0358663

        trajectory = []
        for idx in range(start, end+1):
            file = f"episode_{idx:07d}"
            tmp = np.load(os.path.join(self.calvin_basedir, f'{file}.npz'), allow_pickle=True)
            trajectory.append(tmp)


  

        obs = []
        instrs = []
        actions = []
        state_obs = []
      
        for step in trajectory:
            
            obs.append(step['rgb_static'])
            instruction = language
            instrs.append(instruction)
            action = step['rel_actions']
            result = '[STARTACTION]'
            for idx in range(6):
                tmp =  self.get_bin_id(action[idx], self.action_min[idx], self.action_max[idx], self.calvin_bin_size)
                result += f'[ROBOTACTION{idx}_{tmp}]'
            if action[6] == 1:
                result += '[GRIPPER_OPEN]'
            else:
                result += '[GRIPPER_CLOSE]'
            result += '[ENDOFACTION]'
            actions.append(result)

            state = step['robot_obs']
            result = '[STARTSTATE]'
            for idx in range(14):
                tmp =  self.get_bin_id(state[idx], self.robot_obs_mean[idx] - 3 * self.robot_obs_std[idx], 
                                        self.robot_obs_mean[idx] + 3 * self.robot_obs_std[idx], self.calvin_bin_size)
                result += f'[ROBOTSTATE{idx}_{tmp}]'
            
            if state[14] == 1:
                result += '[GRIPPER_OPENED]'
            else:
                result += '[GRIPPER_CLOSED]'

            result += '[ENDOFSTATE]'
            state_obs.append(result)
                # result += f'[ROBOTACTION{idx}_{tmp}]'
            

      
    
        return {
            "observations": obs,
            'state_obs': state_obs,
            "instructions": instrs, 
            "action": actions,
            "effector_target_translation": None,
            "effector_translation": None,
            "type": 'calvin'
        }
class CalvinDatasetAMLTEval(BaseDataset):
    """

    """
    def __init__(self, finetune=True):
        self.calvin_basedir = '/mnt/calvin/training'
        # self.calvin_basedir = '/home/nikepupu/Desktop/calvin/dataset/calvin_debug_dataset/training'
        self.annotation = np.load(os.path.join(self.calvin_basedir, 'lang_annotations', 'auto_lang_ann.npy'), allow_pickle=True).item()
        with open(os.path.join(self.calvin_basedir,  'statistics.yaml'), 'r') as file:
            self.statistics = yaml.load(file, Loader=yaml.FullLoader)
        
        self.robot_obs_mean = self.statistics['robot_obs'][0]['mean']
        self.robot_obs_std = self.statistics['robot_obs'][0]['std']

        self.language =  self.annotation['language']['ann']
       
        self.annotated_episodis = self.annotation['info']['indx']

        def split_into_segments(ranges, languages):
            split_ranges = []
            for (start, end), lan in zip(ranges, languages):
                while start <= end:
                    # Calculate new end ensuring the range is at most 9 units
                    new_end = min(start + 8, end)
                    # Append the new segment
                    split_ranges.append((start, new_end, lan))
                    # Update start for next segment
                    start = new_end + 1
            return split_ranges
        
        self.annotated_episodis = split_into_segments(self.annotated_episodis, self.language)
        total = len(self.annotated_episodis)
        self.annotated_episodis = self.annotated_episodis[int(total * 0.9):]

        self.calvin_bin_size = 20
        
        bin_sizes = {
            'language_table': 20,
            'calvin': self.calvin_bin_size
        }
        self.base_model_name = "facebook/opt-125m"
        self.tokenizer = init_tokenizer(self.base_model_name, bin_sizes=bin_sizes)

        self.finetune=finetune
        
        
    def __len__(self):
        return len(self.annotated_episodis)
    
    def prepare_calvin_samples(self, calvin_samples):
        obs_batch = []
        instrs_batch = []
        actions_batch = []
        robot_states = []
        m_obs = 4
        
        for sample in calvin_samples:            
            obs =  pad_tensor(torch.tensor(np.array(sample['observations'])).unsqueeze(0), m_obs)
            obs_batch.append(obs)
            instr = sample['instructions'][0]
            instrs_batch.append(instr)
            actions = sample['action']
            robot_states.append(sample['state_obs'])
            
            actions_batch.append(actions)


        for batch in actions_batch:
            while len(batch) < m_obs:
                batch.append('[STARTACTION][TERMINAL][TERMINAL][ENDOFACTION]')
        
        for batch in robot_states:
            while len(batch) < m_obs:
                batch.append(batch[-1])
        
  
        
        obs_batch = torch.cat(obs_batch, dim=0)
        obs_batch = obs_batch.permute(0, 1, 4, 2, 3)
        
        instrs_batch = self.tokenizer(instrs_batch, padding='longest', return_tensors="pt", 
                                      truncation=True, max_length=200, add_special_tokens = False)
        instrs_batch = instrs_batch.input_ids
        max_len = max([len(action) for action in actions_batch])
        for action_lst in actions_batch:
            action_lst.extend(['[ENDOFACTION]'] * (max_len - len(action_lst)))



        a_batch = []
        for action_batch in actions_batch:
            tmp = self.tokenizer(action_batch, padding='longest', return_tensors="pt", truncation=True, max_length=200,  add_special_tokens = False)
            tmp = tmp.input_ids
            a_batch.append(tmp)

        state_batch = []
       
        for action_batch in robot_states:
            tmp = self.tokenizer(action_batch, padding='longest', return_tensors="pt", truncation=True, max_length=200,  add_special_tokens = False)
            tmp = tmp.input_ids
            state_batch.append(tmp)
        

        a_batch = torch.stack(a_batch, dim=0) 
        # print(obs_batch)
        # exit()
        obs_batch = obs_batch.float() / 255



        obs_batch = torch.stack([
            torch.stack([
                self.transforms(frame) for frame in video_sequence
            ]) for video_sequence in obs_batch
        ])

        state_batch = torch.stack(state_batch, dim=0) 
       
        return {
            'video':  obs_batch, 
            'instructions': instrs_batch, 
            'actions': a_batch,
            'robot_state': state_batch,
        }

    
    def collater(self, samples):
        # print('start collate')

        calvin_samples = []
        # print('samples in collate:  ', samples)
        for sample in samples:
            # print(sample.keys())
            del sample['effector_target_translation']
            del sample['effector_translation']
            calvin_samples.append(sample)
         
        
        if len(calvin_samples) > 0:
            
            calvin_return = self.prepare_calvin_samples(calvin_samples)
        else:
            calvin_return = None

        # print(minecraft_return)
        # print(robot_return)
        return {
            'calvin': calvin_return,
            'finetune': self.finetune
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
        start, end, language = self.annotated_episodis[index]
        # format start index to something like this: episode_0358663

        trajectory = []
        for idx in range(start, end+1):
            file = f"episode_{idx:07d}"
            tmp = np.load(os.path.join(self.calvin_basedir, f'{file}.npz'), allow_pickle=True)
            trajectory.append(tmp)


  

        obs = []
        instrs = []
        actions = []
        state_obs = []
      
        for step in trajectory:
            
            obs.append(step['rgb_static'])
            instruction = language
            instrs.append(instruction)
            action = step['rel_actions']
            result = '[STARTACTION]'
            for idx in range(6):
                tmp =  self.get_bin_id(action[idx], -1, 1, self.calvin_bin_size)
                result += f'[ROBOTACTION{idx}_{tmp}]'
            if action[6] == 1:
                result += '[GRIPPER_OPEN]'
            else:
                result += '[GRIPPER_CLOSE]'
            result += '[ENDOFACTION]'
            actions.append(result)

            state = step['robot_obs']
            result = '[STARTSTATE]'
            for idx in range(14):
                tmp =  self.get_bin_id(state[idx], self.robot_obs_mean[idx] - 3 * self.robot_obs_std[idx], 
                                        self.robot_obs_mean[idx] + 3 * self.robot_obs_std[idx], self.calvin_bin_size)
                result += f'[ROBOTSTATE{idx}_{tmp}]'
            
            if state[14] == 1:
                result += '[GRIPPER_OPENED]'
            else:
                result += '[GRIPPER_CLOSED]'

            result += '[ENDOFSTATE]'
            state_obs.append(result)
                # result += f'[ROBOTACTION{idx}_{tmp}]'
            

      
    
        return {
            "observations": obs,
            'state_obs': state_obs,
            "instructions": instrs, 
            "action": actions,
            "effector_target_translation": None,
            "effector_translation": None,
            "type": 'calvin'
        }