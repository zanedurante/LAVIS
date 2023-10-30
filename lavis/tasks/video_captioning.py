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
from lavis.datasets.data_utils import prepare_sample
from lavis.common.logger import MetricLogger
# TODO: Add support for COCO style evaluation for video captions
# TODO: Add video caption dataset completely -- right now it uses a work around to prevent downloading by
# using the coco dataset config
@registry.register_task("video_captioning")
class VideoCaptionTask(BaseTask):
    def __init__(self):
        super().__init__()
    
    def evaluation(self, model, data_loader, cuda_enabled=True):
        metric_logger = MetricLogger(delimiter="  ")
        header = "Evaluation"
        # TODO make it configurable
        print_freq = 50

        results = []
        # import pdb; pdb.set_trace()
        # for samples in metric_logger.log_every(data_loader, print_freq, header):
        #     samples = prepare_sample(samples, cuda_enabled=cuda_enabled)

        #     eval_output = self.valid_step(model=model, samples=samples)
        #     results.extend(eval_output)
        iters_per_epoch = len(data_loader)
        for i in metric_logger.log_every(range(iters_per_epoch), print_freq, header):
            # if using iter-based runner, we stop after iters_per_epoch iterations.
            if i >= iters_per_epoch:
                break
            samples = next(data_loader)
            samples = prepare_sample(samples, cuda_enabled=cuda_enabled)
            eval_output = self.valid_step(model=model, samples=samples)
       
            results.extend(eval_output)

        # if is_dist_avail_and_initialized():
        #     dist.barrier()
        # import pdb; pdb.set_trace()
        return results
    
    def valid_step(self, model, samples):
        # {"loss": loss, "pred": pred, "image": image}
        output = model(samples)
        loss_dict = {}
        for k,v in output.items():
            if "loss" in k or "pred" in k or "image" in k:
                loss_dict[k] = v

        return output["loss"], loss_dict