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
import numpy as np
from PIL import Image  # using pillow-simd for increased speed

import torch
from torchvision import transforms

from skimage.filters import gaussian

import cv2
cv2.setNumThreads(0)

from .base_dataset import BaseDataset
from .warp_dataset import WarpDataset


class KITTIStereoDataset(BaseDataset):

    def __init__(self,
                 data_path,
                 filenames,
                 height,
                 width,
                 is_train,
                 kitti2012=False,
                 disable_normalisation=False,
                 load_gt=True,
                 **kwargs):

        super(KITTIStereoDataset, self).__init__(data_path, filenames, height, width,
                                                 is_train=is_train,
                                                 has_gt=load_gt,
                                                 disable_normalisation=disable_normalisation)

        self.kitti2012 = kitti2012
        self.load_gt = load_gt
        self.img_resizer = transforms.Resize(size=(height, width))

        # We need to specify augmentations differently in newer versions of torchvision.
        # We first try the newer tuple version; if this fails we fall back to scalars
        try:
            self.stereo_brightness = (0.8, 1.2)
            self.stereo_contrast = (0.8, 1.2)
            self.stereo_saturation = (0.8, 1.2)
            self.stereo_hue = (-0.01, 0.01)
            transforms.ColorJitter.get_params(
                self.stereo_brightness, self.stereo_contrast, self.stereo_saturation,
                self.stereo_hue)
        except TypeError:
            self.stereo_brightness = 0.2
            self.stereo_contrast = 0.2
            self.stereo_saturation = 0.2
            self.stereo_hue = 0.01

    def load_images(self, idx, do_flip=False):

        image, stereo_image = self.filenames[idx].split()
        image = self.loader(os.path.join(self.data_path, image))
        stereo_image = self.loader(os.path.join(self.data_path, stereo_image))

        return image, stereo_image

    def load_disparity(self, idx, do_flip=False):

        if self.kitti2012:
            name, _ = self.filenames[idx].split()
            name = name.replace('colored_0', 'disp_occ')
            # name = name.replace('colored_0', 'disp_noc')
            disparity = np.array(Image.open(os.path.join(self.data_path, name))).astype(
                float) / 256

        else:
            name, _ = self.filenames[idx].split()
            name = name.replace('image_2', 'disp_occ_0')
            # name = name.replace('image_2', 'disp_noc_0')
            disparity = np.array(Image.open(os.path.join(self.data_path, name))).astype(float) / 256

        return disparity

    def augment_image(self, image):

        image = np.array(image).astype(float)

        # add some noise to stereo image
        noise = np.random.randn(self.feed_height, self.feed_width, 3) / 50
        image = np.clip(image / 255 + noise, 0, 1) * 255

        # add blurring
        if random.random() > 0.5:
            image = gaussian(image,
                             sigma=random.random(),
                             multichannel=True)

        image = np.clip(image, 0, 255)

        # color augmentation
        stereo_aug = transforms.ColorJitter.get_params(
            self.stereo_brightness, self.stereo_contrast, self.stereo_saturation,
            self.stereo_hue)

        image = stereo_aug(Image.fromarray(image.astype(np.uint8)))

        return image

    def __getitem__(self, idx):

        inputs = {}

        image, stereo_image = self.load_images(idx, do_flip=False)
        if self.load_gt:
            disparity = self.load_disparity(idx)
        else:
            img_width, img_height = image.size
            disparity = np.zeros((img_height, img_width))
        inputs['disparity'] = disparity

        if self.is_train:
            # we are finetuning - crop everything
            height, width, _ = np.array(image).shape
            top = int(random.random() * (height - self.feed_height))
            left = int(random.random() * (width - self.feed_width))
            right, bottom = left + self.feed_width, top + self.feed_height

            image = image.crop((left, top, right, bottom))
            stereo_image = stereo_image.crop((left, top, right, bottom))
            inputs['disparity'] = inputs['disparity'][top:bottom, left:right]

            # now do separate colour augmentation
            image = self.augment_image(image)
            stereo_image = self.augment_image(stereo_image)

        else:
            image = self.img_resizer(image)
            stereo_image = self.img_resizer(stereo_image)

        inputs['image'] = image
        inputs['stereo_image'] = stereo_image
        self.preprocess(inputs)

        return inputs
