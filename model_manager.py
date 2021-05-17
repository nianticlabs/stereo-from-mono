# Copyright Niantic 2020. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Stereo-from-mono licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

import os
from collections import defaultdict
import json
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
from tensorboardX import SummaryWriter

import networks


class ModelManager:

    def __init__(self, opt):

        self.network = opt.network
        self.max_disparity = opt.max_disparity
        self.learning_rate = opt.lr
        self.lr_step_size = opt.lr_step_size

        self.save_folder = os.path.join(opt.log_path, opt.model_name, 'models')
        os.makedirs(self.save_folder, exist_ok=True)
        self.use_cuda = torch.cuda.is_available()

        # build network
        if self.network != 'hourglass':
            raise NotImplementedError('Currently only hourglass network implemented!')
        self.model = networks.hourglass(self.max_disparity,
                                        psm_no_SPP=opt.psm_no_SPP,
                                        big_SPP=opt.big_SPP)
        self.scales = self.model.scales

        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            self.model = nn.DataParallel(self.model)
            self.model.cuda()
        elif self.use_cuda:
            self.model.cuda()
        else:
            print('Not using GPU - is this really what you want?')

        if opt.mode == 'train':
            self.optimiser = torch.optim.Adam(self.model.parameters(),
                                              lr=self.learning_rate)
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimiser,
                                                             step_size=self.lr_step_size)
            self.save_opts(opt)

            print('learning rate {}'.format(self.learning_rate))
            for group in self.optimiser.param_groups:
                print('learning rate {}'.format(group['lr']))
            print('learning rate {}'.format(self.scheduler.get_lr()[0]))

    def load_model(self, weights_path, load_optimiser=False):

        print('loading model weights from {}...'.format(weights_path))
        weights = torch.load(os.path.join(weights_path, 'model.pth'))

        if torch.cuda.is_available():
            try:
                self.model.load_state_dict(weights)
            except RuntimeError:
                new_weights = {}
                for key, val in weights.items():
                    new_key = key[7:]  # remove 'module.' from the start (from multi gpu -> single)
                    new_weights[new_key] = val
                weights = new_weights
                self.model.load_state_dict(weights)
        else:
            self.model.load_state_dict(weights, map_location='cpu')
        print('successfully loaded weights!')

        if load_optimiser:
            print('loading optimiser...')
            weights = torch.load(os.path.join(weights_path, 'optimiser.pth'))
            self.optimiser.load_state_dict(weights)
            print('successfully loaded optimiser!')

            for group in self.optimiser.param_groups:
                print('learning rate {}'.format(group['lr']))

    def save_model(self, folder_name):

        save_path = os.path.join(self.save_folder, folder_name)
        print('saving weights to {}...'.format(save_path))
        os.makedirs(save_path, exist_ok=True)

        torch.save(self.model.state_dict(), os.path.join(save_path,
                                                         'model.pth'))
        torch.save(self.optimiser.state_dict(), os.path.join(save_path,
                                                             'optimiser.pth'))
        print('success!')

    def save_opts(self, opt):

        options = opt.__dict__.copy()

        with open(os.path.join(self.save_folder, 'opts.json'), 'w') as fh:
            json.dump(options, fh, indent=2)
