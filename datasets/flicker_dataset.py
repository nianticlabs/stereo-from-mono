# Copyright Niantic 2020. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Stereo-from-mono licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import os
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import random
import torch
import numpy as np
from PIL import Image  # using pillow-simd for increased speed

from torchvision import transforms

import cv2
cv2.setNumThreads(0)

from .base_dataset import BaseDataset


class FlickerDataset(BaseDataset):

    def __init__(self,
                 data_path,
                 filenames,
                 height,
                 width,
                 is_train,
                 disable_normalisation=False,
                 **kwargs):

        super(FlickerDataset, self).__init__(data_path, filenames, height, width,
                                                 is_train=is_train, has_gt=False,
                                                 disable_normalisation=disable_normalisation)

        self.img_resizer = transforms.Resize(size=(height, width))

    def load_images(self, idx, do_flip=False):
        folder, name = self.filenames[idx].split()
        image = self.loader(os.path.join(self.data_path, folder, '{}_L.png'.format(name)))
        stereo_image = self.loader(os.path.join(self.data_path, folder, '{}_R.png'.format(name)))

        return image, stereo_image

    def load_disparity(self, idx, do_flip=False):
        # just return array of zeros for the original image shape
        folder, name = self.filenames[idx].split()
        image = self.loader(os.path.join(self.data_path, folder, '{}_L.png'.format(name)))
        return np.zeros_like(image)[..., 0]

    def __getitem__(self, idx):

        inputs = {}

        image, stereo_image = self.load_images(idx, do_flip=False)
        disparity = self.load_disparity(idx)
        image = self.img_resizer(image)
        stereo_image = self.img_resizer(stereo_image)

        inputs['image'] = image
        inputs['stereo_image'] = stereo_image
        inputs['disparity'] = torch.from_numpy(disparity).float()
        self.preprocess(inputs)

        return inputs