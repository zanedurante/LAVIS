
# from lavis.datasets.datasets.base_dataset import BaseDataset
import os
import numpy as np
import torch
import torch.nn.functional as F
from lavis.datasets.datasets.base_dataset import BaseDataset
from transformers import AutoTokenizer
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
class LanguageTableDataset(BaseDataset):
    """

    """
    def __init__(self):
        self.basedir = '/home/nikepupu/dataset/language_table'
        # load all npz files under the directory
        # self.episodes = []
        self.files = []
        for file in os.listdir(self.basedir):
            if file.endswith('.npz'):
                self.files.append(os.path.join(self.basedir, file))
                
        
        self.tokenizer =  AutoTokenizer.from_pretrained("facebook/opt-125m")
        for i in range(100):
            self.tokenizer.add_tokens([f"[ROBOTACTIONX{i}]", f"[ROBOTACTIONY{i}]"])
        
        self.tokenizer.add_tokens(['[ENDOFACTION]'])
        self.tokenizer.add_tokens(['[TERMINAL]'])

        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.transforms = self.get_transforms()
        
        
    def __len__(self):
        return len(self.files)
    
    def collater(self, samples):
        obs_batch = []
        instrs_batch = []
        actions_batch = []
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
            # print('actions: ', actions)
            
            actions_batch.append(actions)

        obs_batch = torch.cat(obs_batch, dim=0)
        obs_batch = obs_batch.permute(0, 1, 4, 2, 3)
        
        instrs_batch = self.tokenizer(instrs_batch, padding='longest', return_tensors="pt", 
                                      truncation=True, max_length=200, add_special_tokens = True)
        instrs_batch = instrs_batch.input_ids
        max_len = max([len(action) for action in actions_batch])
        for action_lst in actions_batch:
            action_lst.extend(['[TERMINAL]'] * (max_len - len(action_lst)))
        # print(actions_batch)
        # actions_batch = self.tokenizer(actions_batch, padding='longest', return_tensors="pt", truncation=True, max_length=200)
        # print('action batch')
        # print(actions_batch)
        a_batch = []
        for action_batch in actions_batch:
            tmp = self.tokenizer(action_batch, padding='longest', return_tensors="pt", truncation=True, max_length=200)
            tmp = tmp.input_ids[:,1:-1]
            a_batch.append(tmp)

        a_batch = torch.stack(a_batch, dim=0) 
        obs_batch = obs_batch.float() / 255

        obs_batch = torch.stack([
            torch.stack([
                self.transforms(frame) for frame in video_sequence
            ]) for video_sequence in obs_batch
        ])
        # print(a_batch.shape) 
        # print(obs_batch.shape)
        # print(instrs_batch)
        # exit()
        return {
            'video':  obs_batch, 
            'instructions': instrs_batch, 
            'actions': a_batch
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
            raise ValueError("Number is out of the specified range.")

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
        
        for step in trajectory:
            
            obs.append(step['observation'])
            instruction = step['instruction'] 
            instruction = ''.join(chr(id) for id in instruction)
            instrs.append(step['instruction'])
            if not step['is_terminal']:
                first_dim = self.get_bin_id(step['action'][0], -0.3, 0.3, 100)
                second_dim = self.get_bin_id(step['action'][1], -0.3, 0.3, 100)
                actions.append(f"[ROBOTACTIONX{first_dim}][ROBOTACTIONY{second_dim}][ENDOFACTION]")
            else:
                actions.append('[TERMINAL][TERMINAL][ENDOFACTION]')

        # return obs, instrs, actions, is_firsts, is_lasts, is_terminals
    
        return {
            "observations": obs,
            "instructions": instrs, 
            "action": actions,
        }



if __name__ == "__main__":
    ds = LanguageTableDataset()
    print(len(ds))
    instr = ds[1][1]
    l = len(instr)
    for i in range(l):
        
        output = ''.join(chr(id) for id in instr[i])
        print(output)
    