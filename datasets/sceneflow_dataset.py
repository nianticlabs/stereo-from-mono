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
import webp

import cv2
cv2.setNumThreads(0)

from torchvision import transforms
from skimage.filters import gaussian

from .base_dataset import BaseDataset, read_pfm


def load_webp(image):

    image = webp.load_image(image, 'RGB')

    return image


class SceneFlowDataset(BaseDataset):

    def __init__(self,
                 data_path,
                 filenames,
                 height,
                 width,
                 is_train=False,
                 disable_normalisation=False,
                 disable_synthetic_augmentation=False,
                 **kwargs):

        super(SceneFlowDataset, self).__init__(data_path, filenames, height, width, is_train,
                                               has_gt=True,
                                               disable_normalisation=disable_normalisation)

        self.webp_loader = load_webp  # can't use PIL directly for some reason
        self.disable_synthetic_augmentation = disable_synthetic_augmentation

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

    def crop_all(self, inputs):

        # get crop parameters
        height, width, _ = np.array(inputs['image']).shape
        top = int(random.random() * (height - self.feed_height))
        left = int(random.random() * (width - self.feed_width))
        right, bottom = left + self.feed_width, top + self.feed_height

        for key in ['image', 'stereo_image']:
            inputs[key] = inputs[key].crop((left, top, right, bottom))
        inputs['disparity'] = inputs['disparity'][top:bottom, left:right]

        return inputs

    def augment_stereo_image(self, image):

        if self.disable_synthetic_augmentation:

            return Image.fromarray(image.astype(np.uint8))

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

    def load_data(self, idx, inputs):

        side_lookup = {'left': 'right', 'right': 'left'}

        sceneflow_set = self.filenames[idx].split()[0]

        if sceneflow_set == 'monkaa':

            _, scene, side, frame = self.filenames[idx].split()

            image_path = os.path.join(self.data_path, sceneflow_set,
                                      sceneflow_set + '_frames_finalpass', scene)
            disp_path = os.path.join(self.data_path, sceneflow_set,
                                     sceneflow_set + '_disparities', scene)
            suffix = 'png'

        elif sceneflow_set == 'driving':

            _, focal_length, direction, speed, side, frame = self.filenames[idx].split()

            image_path = os.path.join(self.data_path, sceneflow_set,
                                      sceneflow_set + '_frames_finalpass', focal_length,
                                      direction, speed)
            disp_path = os.path.join(self.data_path, sceneflow_set,
                                     sceneflow_set + '_disparities', focal_length,
                                     direction, speed)
            suffix = 'webp'

        elif sceneflow_set == 'flyingthings3d':

            _, split, scene, seq, side, frame = self.filenames[idx].split()
            disp_path = os.path.join(self.data_path, sceneflow_set,
                                     sceneflow_set + '_disparities', split, scene, seq)

            if self.is_train:
                image_path = os.path.join(self.data_path, sceneflow_set,
                                          sceneflow_set + '_frames_finalpass', split, scene, seq)
                suffix = 'webp'
            else:
                image_path = os.path.join(self.data_path, sceneflow_set,
                                          sceneflow_set + '_frames_cleanpass', split, scene, seq)
                suffix = 'png'

        otherside = side_lookup[side]

        loader = self.webp_loader if suffix == 'webp' else self.loader

        image = loader(os.path.join(image_path, side,
                                        '{}.{}'.format(str(frame).zfill(4), suffix)))
        stereo_image = loader(os.path.join(image_path, otherside,
                                               '{}.{}'.format(str(frame).zfill(4), suffix)))
        disparity = read_pfm(os.path.join(disp_path, side, '{}.pfm'.format(str(frame).zfill(4))))

        if side == 'right':
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            stereo_image = stereo_image.transpose(Image.FLIP_LEFT_RIGHT)
            disparity = np.fliplr(disparity).copy()

        inputs['image'] = image
        inputs['stereo_image'] = stereo_image
        inputs['disparity'] = np.ascontiguousarray(disparity)

    def __getitem__(self, item):

        inputs = {}

        self.load_data(item, inputs)

        if self.is_train:
            inputs = self.crop_all(inputs)
            inputs['stereo_image'] = self.augment_stereo_image(np.array(inputs['stereo_image']))
        else:
            inputs['image'] = self.img_resizer(inputs['image'])
            inputs['stereo_image'] = self.img_resizer(inputs['stereo_image'])

        inputs['mono_disparity'] = inputs['disparity']
        self.preprocess(inputs)

        return inputs
