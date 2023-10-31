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
from lavis.datasets.datasets.webvid_trio_dataset import make_trio_csv_from_original, WebVidCaptionDataset, WebVidCaptionEvalDataset


class RewrittenCaptionDataset(WebVidCaptionDataset):
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
        self.orig_csv_path = "/mnt/datasets_mnt/webvid10m/metadata/rewrite_10M_train.csv"
        self.converted_csv_path = "/mnt/datasets_mnt/webvid10m/metadata/rewrite_10M_train_trio_format.csv"
        super().__init__(vis_processor, text_processor, vis_root, ann_paths, num_skip_frames, total_num_frames, prompt_type)



class RewrittenCaptionEvalDataset(WebVidCaptionEvalDataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths, num_skip_frames=None, total_num_frames=4, prompt_type="image"):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        split (string): val or test
        """
        self.root_dataset_path = "/mnt/datasets_mnt/webvid10m/"
        self.orig_csv_path = "/mnt/datasets_mnt/webvid10m/metadata/rewrite_10M_val.csv"
        self.converted_csv_path = "/mnt/datasets_mnt/webvid10m/metadata/rewrite_10M_val_trio_format.csv"
        super().__init__(vis_processor, text_processor, vis_root, ann_paths, num_skip_frames, total_num_frames, prompt_type) # Note, we keep vis_processor here for compatibility with the original code

