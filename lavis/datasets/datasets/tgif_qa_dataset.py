
import os
from lavis.datasets.datasets.base_dataset import BaseDataset

from lavis.datasets.datasets.caption_datasets import CaptionDataset
from lavis.datasets.datasets.video_caption_datasets import VideoCaptionDataset, VideoCaptionEvalDataset
from lavis.datasets.datasets.trio_video_qa_dataset import TrioVideoQADataset, TrioVideoQAEvalDataset
import decord
import numpy as np
from abc import abstractmethod
from torchvision import transforms
from PIL import Image
import pandas as pd


class TGIFQADataset(TrioVideoQADataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths, num_skip_frames=-1, total_num_frames=4, prompt_type="image"):
        """
        TrioVideoCaptionDataset structure supports fixed FPS and random frame sampling during training through an interface.
        Use num_skip_frames to set the number of frames to skip for fixed FPS sampling. If -1, assumes random frame sampling. If 0, uses every frame, if 1 uses every other frame, etc.
        total_num_frames is the total number of frames to sample from a video for each clip during training. This is equivalent to the number of frames in a video during inference.
        For _load_annotations, you need to either load a CSV file or create a pd.DataFrame with the following structure:
            video, question, answer, start_frame (optional), end_frame (optional)
        split (string): val or test
        """
        self.csv_path = "/mnt/datasets_mnt/tgif/Train_trio_no_count.csv" # Remove count type of question, since we only see 4 frames
        super().__init__(vis_processor, text_processor, vis_root, ann_paths, num_skip_frames, total_num_frames, prompt_type)

    def _load_metadata(self):
        self.metadata = pd.read_csv(self.csv_path)
        return


class TGIFQAEvalDataset(TrioVideoQAEvalDataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths, num_skip_frames=-1, total_num_frames=4, prompt_type="image"):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        split (string): val or test
        """
        self.csv_path = "/mnt/datasets_mnt/tgif/Test_trio.csv"
        super().__init__(vis_processor, text_processor, vis_root, ann_paths, num_skip_frames, total_num_frames) # Note, we keep vis_processor here for compatibility with the original code

    def _load_metadata(self):
        self.metadata = pd.read_csv(self.csv_path)
        return