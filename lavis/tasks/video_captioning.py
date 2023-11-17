"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import json
import os
import torch
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

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
    
    def valid_step(self, model, samples):
        print('run generate')
        output = model.generate(samples)
        print('run generate done')
        return output
    
    def evaluation(self, model, data_loader, cuda_enabled=True):
        def show_image(image, title=''):
            # imagenet_mean = np.array([0.485, 0.456, 0.406])
            # imagenet_std = np.array([0.229, 0.224, 0.225])
            imagenet_mean = np.array([0.48145466, 0.4578275, 0.40821073 ])
            imagenet_std =  np.array([0.26862954, 0.26130258, 0.27577711])
            # image is [H, W, 3]
            assert image.shape[2] == 3
            plt.imshow(torch.clip((image * imagenet_std + imagenet_mean) * 255, 0, 255).int())
            plt.title(title, fontsize=16)
            plt.axis('off')
            return

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
            print(samples['text_output'])
            if i == 0:
                continue
            samples = prepare_sample(samples, cuda_enabled=cuda_enabled)
            print('start valid step')
            # eval_output = self.valid_step(model=model, samples=samples)
            output = model.generate(samples)
            print('valid done')
            exit()
            pred  = eval_output[0]["pred"]
            image = eval_output[0]["image"]
            mask = eval_output[0]["mask"]
            # import pdb; pdb.set_trace()
            mask = mask.unsqueeze(-1).repeat(1, 1, model.visual_encoder.patch_embed.patch_size[0]**2 *3)  # (N, H*W, p*p*3)

            mask = model.visual_encoder.unpatchify(mask)  # 1 is removing, 0 is keeping
            mask = torch.einsum('nfchw->nfhwc', mask).detach().cpu()
            mask = mask[0]

            # y = model.visual_encoder.unpatchify(pred)
            # y = torch.einsum('nchw->nhwc', y).detach().cpu()
            # show_image(y[0], "reconstruction")
            # plt.show()
            # yy = torch.einsum('nchw->nhwc', image).detach().cpu()
            # show_image(yy[0], "original")
            # plt.show()


            y = model.visual_encoder.unpatchify(pred)
            y = torch.einsum('nfchw->nfhwc', y).detach().cpu()
            y = y[0]
            # show_image(y[0][0], "reconstruction")
            # plt.show()
            yy = torch.einsum('nfchw->nfhwc', image).detach().cpu()
            x = yy[0]
            # show_image(yy[0][0], "original")
            # plt.show()

              # masked image
            # import pdb; pdb.set_trace()
            im_masked = x * (1 - mask)

            # MAE reconstruction pasted with visible patches
            im_paste = x * (1 - mask) + y * mask

            # make the plt figure larger
            plt.rcParams['figure.figsize'] = [24, 24]

            plt.subplot(1, 4, 1)
            show_image(x[0], "original")

            plt.subplot(1, 4, 2)
            show_image(im_masked[0], "masked")

            plt.subplot(1, 4, 3)
            show_image(y[0], "reconstruction")

            plt.subplot(1, 4, 4)
            show_image(im_paste[0], "reconstruction + visible")

            plt.show()
            
           
       
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

        return output, loss_dict