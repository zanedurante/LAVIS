
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
class LanguageTableDatasetAMLTTrain(BaseDataset):
    """

    """
    def __init__(self, finetune=False):
        self.basedir = '/mnt/languagetablesim'
        # load all npz files under the directory
        # self.episodes = []
        with open(os.path.join(self.basedir, 'robot.txt'), 'r') as f:
            self.files = f.read().splitlines() 
        self.files = sorted(self.files)
 
        self.base_model_name = "facebook/opt-125m"
        bin_sizes = {
            'language_table': 100,
            'calvin': 100
        }
        self.tokenizer =  init_tokenizer(bin_sizes=bin_sizes, base_model_name=self.base_model_name)
        

        self.transforms = self.get_transforms()
        self.finetune = finetune
        
        
    def __len__(self):
        return len(self.files)
    
    def collater(self, samples):
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
 
        for sample in samples:            
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
        return { 
            'robot':{
                'video':  obs_batch, 
                'instructions': instrs_batch, 
                'actions': a_batch,
                'effector_target_translation': eet_batch,
                'effector_translation': et_batch
                },

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
        file = self.files[index]
        episode = np.load(os.path.join(self.basedir, file), allow_pickle=True)
        trajectory = episode['trajectory']

        obs = []
        instrs = []
        actions = []
        effector_translations = []
        effector_target_translation = [] 
        
        for step in trajectory:
            
            obs.append(step['observation'])
            instruction = step['instruction'] 
            instruction = ''.join(chr(id) for id in instruction)
            instrs.append(step['instruction'])
            ee_t_first_dim =  int(self.get_bin_id(step['effector_translation'][0], 0.15, 0.6, 100))
            ee_t_second_dim = int(self.get_bin_id(step['effector_translation'][1], -0.3, 0.3, 100))
            effector_translations.append(f"[STARTEET][ROBOTEETX{ee_t_first_dim}][ROBOTEETY{ee_t_second_dim}][ENDOFEET]")

            ee_tt_first_dim =  int(self.get_bin_id(step['effector_target_translation'][0], 0.15, 0.6, 100))
            ee_tt_second_dim = int(self.get_bin_id(step['effector_target_translation'][1], -0.3, 0.3, 100))
            effector_target_translation.append(f"[STARTEETT][ROBOTEETTX{ee_tt_first_dim}][ROBOTEETTY{ee_tt_second_dim}][ENDOFEETT]")

            if not step['is_terminal']:
                first_dim =  int(self.get_bin_id(step['action'][0], -0.03, 0.03, 100))
                second_dim = int(self.get_bin_id(step['action'][1], -0.03, 0.03, 100))
                actions.append(f"[STARTACTION][ROBOTACTIONX{first_dim}][ROBOTACTIONY{second_dim}][ENDOFACTION]")
            else:
                actions.append('[STARTACTION][TERMINAL][TERMINAL][ENDOFACTION]')

        # return obs, instrs, actions, is_firsts, is_lasts, is_terminals
    
        return {
            "observations": obs,
            "instructions": instrs, 
            "action": actions,
            "effector_translation": effector_translations,
            "effector_target_translation": effector_target_translation
        }


class LanguageTableDatasetAMLTEval(BaseDataset):
    """

    """
    def __init__(self, finetune=False):
        self.basedir = '/mnt/languagetablesim'
        # load all npz files under the directory
        # self.episodes = []
        with open(os.path.join(self.basedir, 'robot.txt'), 'r') as f:
            self.files = f.read().splitlines() 
        self.files = sorted(self.files)
        total = len(self.files)
        self.files = self.files[int(total * 0.9):]
                
        self.base_model_name = "facebook/opt-125m"
        self.tokenizer =  AutoTokenizer.from_pretrained(self.base_model_name)
        for i in range(21):
            self.tokenizer.add_tokens([f"[ROBOTACTIONX{i}]", f"[ROBOTACTIONY{i}]"])
            self.tokenizer.add_tokens([f"[ROBOTEETX{i}]", f"[ROBOTEETY{i}]"])
            self.tokenizer.add_tokens([f"[ROBOTEETTX{i}]", f"[ROBOTEETTY{i}]"])
        
        self.tokenizer.add_tokens(['[ENDOFACTION]'])
        self.tokenizer.add_tokens(['[TERMINAL]'])

        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.transforms = self.get_transforms()

        self.finetune = finetune
        
        
    def __len__(self):
        return len(self.files)
    
    def collater(self, samples):
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
 
        for sample in samples:            
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
                                      truncation=True, max_length=200, add_special_tokens = True)
        instrs_batch = instrs_batch.input_ids
        max_len = max([len(action) for action in actions_batch])
        for action_lst in actions_batch:
            action_lst.extend(['[ENDOFACTION]'] * (max_len - len(action_lst)))



        a_batch = []
        for action_batch in actions_batch:
            tmp = self.tokenizer(action_batch, padding='longest', return_tensors="pt", truncation=True, max_length=200)
            tmp = tmp.input_ids
            a_batch.append(tmp)

        eet_batch = []
        for action_batch in effector_target_translation:
            tmp = self.tokenizer(action_batch, padding='longest', return_tensors="pt", truncation=True, max_length=200)
            tmp = tmp.input_ids
            eet_batch.append(tmp)
        
        et_batch = []
        for action_batch in effector_translation:
            tmp = self.tokenizer(action_batch, padding='longest', return_tensors="pt", truncation=True, max_length=200)
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
        return {
            'robot':{
                'video':  obs_batch, 
                'instructions': instrs_batch, 
                'actions': a_batch,
                'effector_target_translation': eet_batch,
                'effector_translation': et_batch
                },
                
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
        file = self.files[index]
        episode = np.load(os.path.join(self.basedir, file), allow_pickle=True)
        trajectory = episode['trajectory']

        obs = []
        instrs = []
        actions = []
        effector_translations = []
        effector_target_translation = [] 
        
        for step in trajectory:
            
            obs.append(step['observation'])
            instruction = step['instruction'] 
            instruction = ''.join(chr(id) for id in instruction)
            instrs.append(step['instruction'])
            ee_t_first_dim =  int(self.get_bin_id(step['effector_translation'][0], 0.15, 0.6, 20))
            ee_t_second_dim = int(self.get_bin_id(step['effector_translation'][1], -0.3, 0.3, 20))
            effector_translations.append(f"[ROBOTEETX{ee_t_first_dim}][ROBOTEETY{ee_t_second_dim}]")

            ee_tt_first_dim =  int(self.get_bin_id(step['effector_target_translation'][0], 0.15, 0.6, 20))
            ee_tt_second_dim = int(self.get_bin_id(step['effector_target_translation'][1], -0.3, 0.3, 20))
            effector_target_translation.append(f"[ROBOTEETTX{ee_tt_first_dim}][ROBOTEETTY{ee_tt_second_dim}]")

            if not step['is_terminal']:
                first_dim =  int(self.get_bin_id(step['action'][0], -0.03, 0.03, 20))
                second_dim = int(self.get_bin_id(step['action'][1], -0.03, 0.03, 20))
                actions.append(f"[ROBOTACTIONX{first_dim}][ROBOTACTIONY{second_dim}][ENDOFACTION]")
            else:
                actions.append('[STARTACTION][TERMINAL][TERMINAL][ENDOFACTION]')

        # return obs, instrs, actions, is_firsts, is_lasts, is_terminals
    
        return {
            "observations": obs,
            "instructions": instrs, 
            "action": actions,
            "effector_translation": effector_translations,
            "effector_target_translation": effector_target_translation
        }


# if __name__ == "__main__":
#     ds = LanguageTableDataset()
#     print(len(ds))
#     instr = ds[1][1]
#     l = len(instr)
#     for i in range(l):
        
#         output = ''.join(chr(id) for id in instr[i])
#         print(output)
    