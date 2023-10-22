"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import os
from collections import OrderedDict

from lavis.datasets.datasets.base_dataset import BaseDataset
from PIL import Image


class __DisplMixin:
    def displ_item(self, index):
        sample, ann = self.__getitem__(index), self.annotation[index]

        return OrderedDict(
            {
                "file": ann["image"],
                "caption": ann["caption"],
                "image": sample["image"],
            }
        )


class CaptionDataset(BaseDataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

        self.img_ids = {}
        n = 0
        for ann in self.annotation:
            img_id = ann["image_id"]
            if img_id not in self.img_ids.keys():
                self.img_ids[img_id] = n
                n += 1
        
        self.prompts = [
            "A short image caption: ",
            "A short image description: ",
            "A photo of ",
            "An image that shows ",
            "Write a short description of the image. ",
            "Write a description for the photo.",
            "Provide a description of what is presented in the photo. ",
            "Briefly describe the content of the image. ",
            "Can you briefly explain what you see in the image? ",
            "Could you use a few words to describe what you perceive in the photo? ",
            "Please provide a short depiction of the picture. ",
            "Using language, provide a short account of the image. ",
            "Use a few words to illustrate what is happening in the picture. ",
        ]
        print("Using prompts: ", self.prompts)
        self.prompt = self.prompts[0]
        self.prompt_idx = 0
    
    def _get_next_prompt(self):
        self.prompt_idx += 1
        self.prompt_idx = self.prompt_idx % len(self.prompts)
        return self.prompts[self.prompt_idx]

    def __getitem__(self, index):

        # TODO this assumes image input, not general enough
        ann = self.annotation[index]

        image_path = os.path.join(self.vis_root, ann["image"])
        image = Image.open(image_path).convert("RGB")

        image = self.vis_processor(image)
        caption = self.text_processor(ann["caption"])

        # Cycle through the prompts
        input_text = self._get_next_prompt()

        return {
            "image": image,
            "text_input": input_text,
            "text_output": caption
            # "image_id": self.img_ids[ann["image_id"]],
        }


class CaptionEvalDataset(BaseDataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        split (string): val or test
        """
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

    def __getitem__(self, index):

        ann = self.annotation[index]

        image_path = os.path.join(self.vis_root, ann["image"])
        image = Image.open(image_path).convert("RGB")
        image = self.vis_processor(image)

        return {
            "image": image,
            "image_id": ann["image_id"],
            "instance_id": ann["instance_id"],
        }
