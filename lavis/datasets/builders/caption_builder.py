"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

from lavis.datasets.builders.base_dataset_builder import BaseDatasetBuilder
from lavis.datasets.datasets.coco_caption_datasets import (
    COCOCapDataset,
    COCOCapEvalDataset,
    NoCapsEvalDataset,
)

from lavis.datasets.datasets.webvid_trio_dataset import (
    WebVidCaptionDataset,
    WebVidCaptionEvalDataset,
)

from lavis.datasets.datasets.minecraft_dataset_amlt import (
    MinecraftVidDatasetAMLT
)

from lavis.datasets.datasets.minecraft_dataset import (
    MinecraftVidDataset
)
from lavis.datasets.datasets.rewritten_trio_dataset import (
    RewrittenCaptionDataset,
    RewrittenCaptionEvalDataset,
)

from lavis.common.registry import registry
from lavis.datasets.datasets.video_caption_datasets import (
    VideoCaptionDataset,
    VideoCaptionEvalDataset,
)
@registry.register_builder("local_minecraft")
class MinecraftBuilderLocal(BaseDatasetBuilder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.train_dataset_cls = MinecraftVidDataset
        self.eval_dataset_cls = MinecraftVidDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/minecraft/local_minecraft.yaml", 
    }

    def build_datasets(self, cfg=None):
        print("Assuming minecraft dataset is already stored -- skipping build!!")
        datasets = {"train": None, "eval": None}

        self.build_processors()
        
        num_frames = self.config.get("total_num_frames", 4)
        num_skip_frames = self.config.get("num_skip_frames", 4)



        for split, dataset_cls in zip(["train", "eval"], [self.train_dataset_cls, self.eval_dataset_cls]):
            is_train = split == "train"
            vis_processor = (
                self.vis_processors["train"]
                if is_train
                else self.vis_processors["eval"]
            )
            text_processor = (
                self.text_processors["train"]
                if is_train
                else self.text_processors["eval"]
            )
            ann_paths = None # Not used for TrioVideo datasets 
            vis_root = None # Not used for TrioVideo datasets
            datasets[split] = dataset_cls(
                        vis_processor=vis_processor,
                        text_processor=text_processor,
                        ann_paths=ann_paths,
                        vis_root=vis_root,
                        num_skip_frames=num_skip_frames,
                        total_num_frames=num_frames,
            )

        return datasets
    

@registry.register_builder("minecraft")
class MinecraftBuilder(BaseDatasetBuilder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.train_dataset_cls = MinecraftVidDatasetAMLT
        self.eval_dataset_cls = MinecraftVidDatasetAMLT

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/minecraft/defaults_action.yaml", 
    }

    def build_datasets(self, cfg=None):
        print("Assuming minecraft dataset is already stored -- skipping build!!")
        datasets = {"train": None, "eval": None}

        self.build_processors()
        
        num_frames = self.config.get("total_num_frames", 4)
        num_skip_frames = self.config.get("num_skip_frames", 4)



        for split, dataset_cls in zip(["train", "eval"], [self.train_dataset_cls, self.eval_dataset_cls]):
            is_train = split == "train"
            vis_processor = (
                self.vis_processors["train"]
                if is_train
                else self.vis_processors["eval"]
            )
            text_processor = (
                self.text_processors["train"]
                if is_train
                else self.text_processors["eval"]
            )
            ann_paths = None # Not used for TrioVideo datasets 
            vis_root = None # Not used for TrioVideo datasets
            scale = self.config.get("scale", "small")
            datasets[split] = dataset_cls(
                        vis_processor=vis_processor,
                        text_processor=text_processor,
                        ann_paths=ann_paths,
                        vis_root=vis_root,
                        num_skip_frames=num_skip_frames,
                        total_num_frames=num_frames,
                        scale=scale
            )

        return datasets

@registry.register_builder("webvid_caption")
class WebVidCapBuilder(BaseDatasetBuilder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.train_dataset_cls = WebVidCaptionDataset
        self.eval_dataset_cls = WebVidCaptionEvalDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/webvid/defaults_cap.yaml", 
    }

    def build_datasets(self, cfg=None):
        print("Assuming WebVid dataset is already stored -- skipping build!!")
        datasets = {"train": None, "eval": None}

        self.build_processors()

        num_frames = self.config.get("total_num_frames", 4)
        num_skip_frames = self.config.get("num_skip_frames", -1)



        for split, dataset_cls in zip(["train", "eval"], [self.train_dataset_cls, self.eval_dataset_cls]):
            is_train = split == "train"
            vis_processor = (
                self.vis_processors["train"]
                if is_train
                else self.vis_processors["eval"]
            )
            text_processor = (
                self.text_processors["train"]
                if is_train
                else self.text_processors["eval"]
            )
            ann_paths = None # Not used for TrioVideo datasets 
            vis_root = None # Not used for TrioVideo datasets
            datasets[split] = dataset_cls(
                        vis_processor=vis_processor,
                        text_processor=text_processor,
                        ann_paths=ann_paths,
                        vis_root=vis_root,
                        num_skip_frames=num_skip_frames,
                        total_num_frames=num_frames,
            )

        return datasets

@registry.register_builder("rewritten_caption")
class RewrittenCapBuilder(WebVidCapBuilder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.train_dataset_cls = RewrittenCaptionDataset
        self.eval_dataset_cls = RewrittenCaptionEvalDataset
    
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/webvid/rewritten_cap.yaml",
    }        

@registry.register_builder("coco_caption")
class COCOCapBuilder(BaseDatasetBuilder):
    train_dataset_cls = COCOCapDataset
    eval_dataset_cls = COCOCapEvalDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/coco/defaults_cap.yaml",
    }


@registry.register_builder("nocaps")
class COCOCapBuilder(BaseDatasetBuilder):
    eval_dataset_cls = NoCapsEvalDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/nocaps/defaults.yaml",
    }


@registry.register_builder("msrvtt_caption")
class MSRVTTCapBuilder(BaseDatasetBuilder):
    train_dataset_cls = VideoCaptionDataset
    eval_dataset_cls = VideoCaptionEvalDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/msrvtt/defaults_cap.yaml",
    }


@registry.register_builder("msvd_caption")
class MSVDCapBuilder(BaseDatasetBuilder):
    train_dataset_cls = VideoCaptionDataset
    eval_dataset_cls = VideoCaptionEvalDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/msvd/defaults_cap.yaml",
    }


@registry.register_builder("vatex_caption")
class VATEXCapBuilder(BaseDatasetBuilder):
    train_dataset_cls = VideoCaptionDataset
    eval_dataset_cls = VideoCaptionEvalDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/vatex/defaults_cap.yaml",
    }
