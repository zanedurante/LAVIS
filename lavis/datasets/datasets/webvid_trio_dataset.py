"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import os
from lavis.datasets.datasets.base_dataset import BaseDataset

from lavis.datasets.datasets.caption_datasets import CaptionDataset
from lavis.datasets.datasets.trio_video_caption_dataset import TrioVideoCaptionDataset, TrioVideoCaptionEvalDataset
import decord
import pandas as pd
from tqdm import tqdm

decord.bridge.set_bridge('torch')

def make_trio_csv_from_original(original_path, trio_path, data_dir="/mnt/datasets_mnt/webvid10m"): 
    # Here data_dir is the root directory of the webvid dataset
    df = pd.read_csv(original_path)
    captions = df["name"].tolist()
    page_dirs = df["page_dir"].tolist()
    video_ids = df["videoid"].tolist()
    # Create a new dataframe with the right format
    def _get_video_path(sample):
        rel_video_fp = os.path.join(sample['page_dir'], str(sample['videoid']) + '.mp4')
        full_video_fp = os.path.join(data_dir, 'videos', rel_video_fp)
        if os.path.exists(full_video_fp):
            return full_video_fp
        return None
    # use tqdm to show progress bar
    video_paths = []
    for page_dir, video_id in tqdm(zip(page_dirs, video_ids), total=len(page_dirs)):
        video_paths.append(_get_video_path({"page_dir":page_dir, "videoid":video_id})) 
    # Filter out videos that don't exist
    prev_len = len(video_paths)
    video_paths, captions = zip(*[(video_path, caption) for video_path, caption in zip(video_paths, captions) if video_path is not None])
    
    print("Filtered out", prev_len - len(video_paths), "videos that don't exist.")

    new_df = pd.DataFrame({"video":video_paths, "caption":captions})
    print("Saving new CSV file in trio format to:", trio_path)
    new_df.to_csv(trio_path, index=False)
    return



class WebVidCaptionDataset(TrioVideoCaptionDataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths, num_skip_frames=None, total_num_frames=4, prompt_type="image"):
        """
        TrioVideoCaptionDataset structure supports fixed FPS and random frame sampling during training through an interface.
        Use num_skip_frames to set the number of frames to skip for fixed FPS sampling. If None, assumes random frame sampling.
        total_num_frames is the total number of frames to sample from a video for each clip during training. This is equivalent to the number of frames in a video during inference.
        For _load_annotations, you need to either load a CSV file or create a pd.DataFrame with the following structure:
            video, caption, start_frame (optional), end_frame (optional)
        split (string): val or test
        """
        self.root_dataset_path = "/mnt/datasets_mnt/webvid10m/"
        self.orig_csv_path = "/mnt/datasets_mnt/webvid10m/metadata/results_10M_train.csv"
        self.converted_csv_path = "/mnt/datasets_mnt/webvid10m/metadata/WebVid_10M_train_trio_format.csv"
        super().__init__(vis_processor, text_processor, vis_root, ann_paths, num_skip_frames, total_num_frames, prompt_type)


    def _load_metadata(self, reload_csv=False):
        """
        Load metadata from a CSV file or generate pd.DataFrame.  Resulting pandas dataframe should have structure:
            video, caption, start_frame (optional), end_frame (optional)
        """


        # Create a new CSV file in the trio format if it doesn't exist
        # Format: "video", "caption", "start_frame", "end_frame", for webvid the last two are 0 and -1 (defaults) so we can ignore them
        if not os.path.exists(self.converted_csv_path) or reload_csv:
            print("Converting original CSV file to trio format...")
            make_trio_csv_from_original(self.orig_csv_path, self.converted_csv_path, self.root_dataset_path)

        self.metadata = pd.read_csv(self.converted_csv_path)
        return


class WebVidCaptionEvalDataset(TrioVideoCaptionEvalDataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths, num_skip_frames=None, total_num_frames=4, prompt_type="image"):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        split (string): val or test
        """
        self.root_dataset_path = "/mnt/datasets_mnt/webvid10m/"
        self.orig_csv_path = "/mnt/datasets_mnt/webvid10m/metadata/results_10M_val.csv"
        self.converted_csv_path = "/mnt/datasets_mnt/webvid10m/metadata/WebVid_10M_val_trio_format.csv"
        super().__init__(vis_processor, text_processor, vis_root, ann_paths, num_skip_frames, total_num_frames, prompt_type) # Note, we keep vis_processor here for compatibility with the original code
        # We use a custome visual processor here that supports FPS sampling + video transforms



    def _load_metadata(self):
        """
        Load metadata from a CSV file or generate pd.DataFrame.  Resulting pandas dataframe should have structure:
            video, caption, start_frame (optional), end_frame (optional)
        """

        # Create a new CSV file in the trio format if it doesn't exist
        # Format: "video", "caption", "start_frame", "end_frame", for webvid the last two are 0 and -1 (defaults) so we can ignore them
        if not os.path.exists(self.converted_csv_path):
            print("Converting original CSV file to trio format...")
            make_trio_csv_from_original(self.orig_csv_path, self.converted_csv_path, self.root_dataset_path)

        self.metadata = pd.read_csv(self.converted_csv_path)
        return