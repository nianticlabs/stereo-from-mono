# Copyright Niantic 2020. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Stereo-from-mono licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

import os
from collections import defaultdict
import json
import time
from collections import defaultdict

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import cv2

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import torch.nn.functional as F

from skimage import io
from datasets import SceneFlowDataset, KITTIStereoDataset, ETH3DStereoDataset, \
    MiddleburyStereoDataset, FlickerDataset
from model_manager import ModelManager
from utils import readlines, load_config

from tqdm import tqdm


data_type_lookup = {
                    'eth3d': ETH3DStereoDataset,
                    'middlebury': MiddleburyStereoDataset,
                    'flicker': FlickerDataset,
                    'kitti2015': KITTIStereoDataset,
                    'kitti2012': KITTIStereoDataset,
                    'kitti2015submission': KITTIStereoDataset,
                    'sceneflow': SceneFlowDataset}


sizes_lookup = {
                'hourglass':{
                    'kitti2015': (1280, 384),
                    'kitti2012': (1280, 384),
                    'eth3d': (768, 448),
                    'middlebury': (1280, 768),
                    'flicker': (736, 1120),
                    'kitti2015submission': (1280, 384),
                    'sceneflow': (960, 512)},
                }


class InferenceManager:
    """
    Main training script called from main.py.

    """

    def __init__(self, options):
        print('---------------')
        self.opt = options
        # Create network and optimiser
        self.model_manager = ModelManager(self.opt)
        assert self.opt.load_path is not None
        self.model_manager.load_model(weights_path=self.opt.load_path, load_optimiser=False)

        # extract model, optimiser and scheduler for easier access
        self.model = self.model_manager.model
        self.model.eval()

        path_info = load_config(self.opt.config_path)

        self.test_loaders = {}
        for test_data_type in self.opt.test_data_types:

            data_path = path_info[test_data_type]
            width, height = sizes_lookup[self.opt.network][test_data_type]

            # create dataloaders
            folder = 'kitti' if 'kitti' in test_data_type else test_data_type
            textfile = test_data_type + '.txt' if 'kitti' in test_data_type else 'test_files.txt'
            filename_path = os.path.join('splits', folder, textfile)
            test_filenames = readlines(filename_path)

            dataset_class = data_type_lookup[test_data_type]
            test_dataset = dataset_class(data_path,
                                         test_filenames, height,
                                         width, is_train=False,
                                         disable_normalisation=self.opt.disable_normalisation,
                                         kitti2012=test_data_type == 'kitti2012',
                                         load_gt=test_data_type != 'kitti2015submission')

            test_loader = DataLoader(test_dataset, shuffle=False, drop_last=False,
                                     num_workers=self.opt.num_workers,
                                     batch_size=1)

            self.test_loaders[test_data_type] = test_loader
        self.error_metrics = defaultdict(list)
        self.resized_disps = []

    def run_inference(self):

        all_errors = {}

        for data_type, loader in self.test_loaders.items():
            print('---------------')
            print('running evaluation on:')
            print(data_type)

            self.error_metrics = defaultdict(list)
            self.resized_disps = []
            with torch.no_grad():
                for inputs in tqdm(loader, ncols=60, position=0, leave=True):
                    _ = self.process_batch(inputs,
                                           compute_errors=data_type not in ['flicker',
                                                                            'kitti2015submission'])

                for key, val in self.error_metrics.items():
                    self.error_metrics[key] = str(np.round(np.mean(val), 5))
                all_errors[data_type] = self.error_metrics

                if self.opt.save_disparities:
                    # also save resized disparities for visualisation
                    _savepath = os.path.join(self.opt.load_path, data_type, 'npys')
                    os.makedirs(_savepath, exist_ok=True)
                    for idx, disp in enumerate(self.resized_disps):
                        np.save(os.path.join(_savepath, '{}.npy'.format(str(idx).zfill(3))), disp)

                    if data_type == 'kitti2015submission':
                        _savepath = os.path.join(_savepath, 'disp_0')
                        os.makedirs(_savepath, exist_ok=True)
                        for idx, disp in enumerate(self.resized_disps):
                            disp = (disp * 256).astype(np.uint16)
                            print(disp.shape)
                            io.imsave(os.path.join(_savepath,
                                                   '{}_10.png'.format(str(idx).zfill(6))), disp)

        print('Finished inference!')
        print('---------------')
        for data_type, errors in all_errors.items():
            print('Metrics for {}:'.format(data_type))
            for key, error in errors.items():
                print('{} -- {}'.format(key, error))
            print('---------------')

        with open(os.path.join(self.opt.load_path, 'eval_results.json'), 'w') as file_handler:
            json.dump(all_errors, file_handler, indent=2)

    def process_batch(self, inputs, compute_errors=True):

        # move to GPU
        if torch.cuda.is_available():
            for key, val in inputs.items():
                inputs[key] = val.cuda()

        outputs = self.model(inputs['image'], inputs['stereo_image'])
        preds = outputs[('raw', 0)][:, 0].cpu().numpy()

        # get errors
        gts = inputs['disparity'].cpu().numpy()
        for i in range(len(gts)):
            # resize and rescale prediction to match gt
            height, width = gts[i].shape
            pred_disp = cv2.resize(preds[i], dsize=(width, height)) * width / preds[i].shape[
                1]

            if compute_errors:
                d1, d2, d3, EPE = self.compute_errors(gts[i], pred_disp)
                self.error_metrics['d1'].append(d1)
                self.error_metrics['d2'].append(d2)
                self.error_metrics['d3'].append(d3)
                self.error_metrics['EPE'].append(EPE)

            if self.opt.save_disparities:
                self.resized_disps.append(pred_disp)

        return outputs

    def compute_errors(self, gt_disp, pred_disp):

        mask = gt_disp > 0
        abs_diff = np.abs(gt_disp[mask] - pred_disp[mask])
        EPE = abs_diff.mean()

        d1 = (abs_diff >= 1).sum() / mask.sum()
        d2 = (abs_diff >= 2).sum() / mask.sum()
        d3 = (abs_diff >= 3).sum() / mask.sum()
        return d1, d2, d3, EPE
