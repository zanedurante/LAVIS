"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import os
from lavis.datasets.datasets.base_dataset import BaseDataset

from lavis.datasets.datasets.caption_datasets import CaptionDataset
from lavis.datasets.datasets.video_caption_dataset import VideoCaptionDataset
import decord

decord.bridge.set_bridge('torch')

class TrioVideoCaptionDataset(VideoCaptionDataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths, num_skip_frames=None, total_num_frames=4):
        """
        TrioVideoCaptionDataset structure supports fixed FPS and random frame sampling during training through an interface.
        Use num_skip_frames to set the number of frames to skip for fixed FPS sampling. If None, assumes random frame sampling.
        total_num_frames is the total number of frames to sample from a video for each clip during training. This is equivalent to the number of frames in a video during inference.
        For _load_annotations, you need to either load a CSV file or create a pd.DataFrame with the following structure:
            video, caption, start_frame (optional), end_frame (optional)
        split (string): val or test
        """
        super().__init__(vis_processor, text_processor, vis_root, ann_paths, num_skip_frames, total_num_frames) # Note, we keep vis_processor here for compatibility with the original code
        # We use a custome visual processor here that supports FPS sampling + video transforms
        self.total_num_frames = total_num_frames
        self.num_skip_frames = num_skip_frames
        self._load_metadata()

    @abstractmethod
    def _load_metadata(self):
        """
        Load metadata from a CSV file or generate pd.DataFrame.  Resulting pandas dataframe should have structure:
            video, caption, start_frame (optional), end_frame (optional)
        """
        self.metadata = None
        pass

    def _load_video(self, video_path, start_frame=0, end_frame=-1):
        frame_indices = None
        video_reader = decord.VideoReader(video_path, num_threads=1)
        video_length = len(video_reader)

        if self.num_skip_frames is None: # Use random frame sampling
            frame_indices = np.random.randint(0, video_length, self.num_frames)
            frame_indices = np.sort(frame_indices)
        if self.num_skip_frames > 0:
            # TODO: Implement variable/random offsets for fixed FPS sampling
            frame_indices = np.arange(start_frame, end_frame, self.num_skip_frames)[:self.total_num_frames]

        frames = video_reader.get_batch(frame_indices)
        frames = frames.float() / 255
        frames = frames.permute(0, 3, 1, 2)
        return frames
            


    def __getitem__(self, index):

        ann = self.metadata[index]

        vname = ann["video"]
        video_path = os.path.join(self.vis_root, vname)
        start_frame = ann.get("start_frame", 0)
        end_frame = ann.get("end_frame", -1)

        video = _load_video(video_path, start_frame, end_frame)
        caption = self.text_processor(ann["caption"])

        # "image_id" is kept to stay compatible with the COCO evaluation format
        return {
            "video": video,
            "text_input": caption,
            "image_id": self.img_ids[ann["image_id"]],
        }


class TrioVideoCaptionEvalDataset(VideoCaptionEvalDataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths, num_skip_frames=None, total_num_frames=4):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        split (string): val or test
        """
        super().__init__(vis_processor, text_processor, vis_root, ann_paths, num_skip_frames, total_num_frames) # Note, we keep vis_processor here for compatibility with the original code
        # We use a custome visual processor here that supports FPS sampling + video transforms
        self.total_num_frames = total_num_frames
        self.num_skip_frames = num_skip_frames
        self._load_metadata()

    @abstractmethod
    def _load_metadata(self):
        """
        Load metadata from a CSV file or generate pd.DataFrame.  Resulting pandas dataframe should have structure:
            video, caption, start_frame (optional), end_frame (optional)
        """
        self.metadata = None
        pass

    def _load_video(self, video_path, start_frame=0, end_frame=-1):
        frame_indices = None
        video_reader = decord.VideoReader(video_path, num_threads=1)
        video_length = len(video_reader)

        if self.num_skip_frames is None: # Use evenly spread frame sampling (for eval datasets)
            frame_indices = np.linspace(0, video_length, self.num_frames)
        if self.num_skip_frames > 0:
            # TODO: Implement variable/random offsets for fixed FPS sampling
            frame_indices = np.arange(start_frame, end_frame, self.num_skip_frames)[:self.total_num_frames]

        frames = video_reader.get_batch(frame_indices)
        frames = frames.float() / 255
        frames = frames.permute(0, 3, 1, 2)
        return frames

    def __getitem__(self, index):

        ann = self.metadata[index]

        vname = ann["video"]
        video_path = os.path.join(self.vis_root, vname)
        start_frame = ann.get("start_frame", 0)
        end_frame = ann.get("end_frame", -1)

        video = _load_video(video_path, start_frame, end_frame)
        caption = self.text_processor(ann["caption"])

        # "image_id" is kept to stay compatible with the COCO evaluation format
        return {
            "video": video,
            "text_input": caption,
            "image_id": self.img_ids[ann["image_id"]],
        }
