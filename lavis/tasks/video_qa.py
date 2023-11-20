"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import json
import os

from lavis.common.dist_utils import main_process
from lavis.common.registry import registry
from lavis.tasks.base_task import BaseTask

# TODO: Add support for COCO style evaluation for video captions
# TODO: Add video caption dataset completely -- right now it uses a work around to prevent downloading by
# using the coco dataset config
@registry.register_task("video_qa")
class VideoQATask(BaseTask):
    def __init__(self):
        super().__init__()