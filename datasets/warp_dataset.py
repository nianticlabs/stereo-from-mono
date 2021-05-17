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
import time

import torch
import torch.utils.data as data
from torchvision import transforms
import torch.nn.functional as F

from skimage.filters import gaussian, sobel
from skimage.color import rgb2grey

from scipy.interpolate import griddata
import cv2
cv2.setNumThreads(0)

from .base_dataset import BaseDataset
from utils import transfer_color


class WarpDataset(BaseDataset):

    def __init__(self,
                 data_path,
                 filenames,
                 feed_height,
                 feed_width,
                 max_disparity,
                 is_train=True,
                 disable_normalisation=False,
                 keep_aspect_ratio=True,
                 disable_synthetic_augmentation=False,
                 disable_sharpening=False,
                 monodepth_model='midas',
                 disable_background=False,
                 **kwargs):

        super(WarpDataset, self).__init__(data_path, filenames, feed_height, feed_width,
                                          is_train=is_train, has_gt=True,
                                          disable_normalisation=disable_normalisation,
                                          keep_aspect_ratio=keep_aspect_ratio)

        self.max_disparity = max_disparity
        self.disable_synthetic_augmentation = disable_synthetic_augmentation
        self.disable_sharpening = disable_sharpening
        self.monodepth_model = monodepth_model
        self.disable_background = disable_background

        # do image generation for a wider image so we can crop off missing pixels
        self.process_width = self.feed_width + self.max_disparity

        self.xs, self.ys = np.meshgrid(np.arange(self.process_width), np.arange(self.feed_height))

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

        self.silly_svsm = False

    def load_images(self, idx, do_flip=False):
        raise NotImplementedError

    def load_disparity(self, idx, do_flip=False):
        raise NotImplementedError

    def process_disparity(self, disparity, max_disparity_range=(40, 196)):
        """ Depth predictions have arbitrary scale - need to convert to a pixel disparity"""

        disparity = disparity.copy()

        # make disparities positive
        min_disp = disparity.min()
        if min_disp < 0:
            disparity += np.abs(min_disp)

        if random.random() < 0.01:
            # make max warped disparity bigger than network max -> will be clipped to max disparity,
            # but will mean network is robust to disparities which are too big
            max_disparity_range = (self.max_disparity * 1.05, self.max_disparity * 1.15)

        disparity /= disparity.max()  # now 0-1

        scaling_factor = (max_disparity_range[0] + random.random() *
                          (max_disparity_range[1] - max_disparity_range[0]))
        disparity *= scaling_factor

        if not self.disable_sharpening:
            # now find disparity gradients and set to nearest - stop flying pixels
            edges = sobel(disparity) > 3
            disparity[edges] = 0
            mask = disparity > 0

            try:
                disparity = griddata(np.stack([self.ys[mask].ravel(), self.xs[mask].ravel()], 1),
                                     disparity[mask].ravel(), np.stack([self.ys.ravel(),
                                                                        self.xs.ravel()], 1),
                                     method='nearest').reshape(self.feed_height, self.process_width)
            except (ValueError, IndexError) as e:
                pass  # just return disparity

        return disparity

    def prepare_sizes(self, inputs):

        height, width, _ = np.array(inputs['left_image']).shape

        if self.keep_aspect_ratio:
            if self.feed_height <= height and self.process_width <= width:
                # can simply crop the image
                target_height = height
                target_width = width

            else:
                # check the constraint
                current_ratio = height / width
                target_ratio = self.feed_height / self.process_width

                if current_ratio < target_ratio:
                    # height is the constraint
                    target_height = self.feed_height
                    target_width = int(self.feed_height / height * width)

                elif current_ratio > target_ratio:
                    # width is the constraint
                    target_height = int(self.process_width / width * height)
                    target_width = self.process_width

                else:
                    # ratio is the same - just resize
                    target_height = self.feed_height
                    target_width = self.process_width

        else:
            target_height = self.feed_height
            target_width = self.process_width

        inputs = self.resize_all(inputs, target_height, target_width)

        # now do cropping
        if target_height == self.feed_height and target_width == self.process_width:
            # we are already at the correct size - no cropping
            pass
        else:
            self.crop_all(inputs)

        return inputs

    def crop_all(self, inputs):

        # get crop parameters
        height, width, _ = np.array(inputs['left_image']).shape
        top = int(random.random() * (height - self.feed_height))
        left = int(random.random() * (width - self.process_width))
        right, bottom = left + self.process_width, top + self.feed_height

        for key in ['left_image', 'background']:
            inputs[key] = inputs[key].crop((left, top, right, bottom))
        inputs['loaded_disparity'] = inputs['loaded_disparity'][top:bottom, left:right]

        return inputs

    @staticmethod
    def resize_all(inputs, height, width):

        # images
        img_resizer = transforms.Resize(size=(height, width))
        for key in ['left_image', 'background']:
            inputs[key] = img_resizer(inputs[key])
        # disparity - needs rescaling
        disp = inputs['loaded_disparity']
        disp *= width / disp.shape[1]

        disp = cv2.resize(disp.astype(float), (width, height))  # ensure disp is float32 for cv2
        inputs['loaded_disparity'] = disp

        return inputs

    def get_occlusion_mask(self, shifted):

        mask_up = shifted > 0
        mask_down = shifted > 0

        shifted_up = np.ceil(shifted)
        shifted_down = np.floor(shifted)

        for col in range(self.process_width - 2):
            loc = shifted[:, col:col + 1]  # keepdims
            loc_up = np.ceil(loc)
            loc_down = np.floor(loc)

            _mask_down = ((shifted_down[:, col + 2:] != loc_down) * (
            (shifted_up[:, col + 2:] != loc_down))).min(-1)
            _mask_up = ((shifted_down[:, col + 2:] != loc_up) * (
            (shifted_up[:, col + 2:] != loc_up))).min(-1)

            mask_up[:, col] = mask_up[:, col] * _mask_up
            mask_down[:, col] = mask_down[:, col] * _mask_down

        mask = mask_up + mask_down
        return mask

    def project_image(self, image, disp_map, background_image):

        image = np.array(image)
        background_image = np.array(background_image)

        # set up for projection
        warped_image = np.zeros_like(image).astype(float)
        warped_image = np.stack([warped_image] * 2, 0)
        pix_locations = self.xs - disp_map

        # find where occlusions are, and remove from disparity map
        mask = self.get_occlusion_mask(pix_locations)
        masked_pix_locations = pix_locations * mask - self.process_width * (1 - mask)

        # do projection - linear interpolate up to 1 pixel away
        weights = np.ones((2, self.feed_height, self.process_width)) * 10000

        for col in range(self.process_width - 1, -1, -1):
            loc = masked_pix_locations[:, col]
            loc_up = np.ceil(loc).astype(int)
            loc_down = np.floor(loc).astype(int)
            weight_up = loc_up - loc
            weight_down = 1 - weight_up

            mask = loc_up >= 0
            mask[mask] = \
                weights[0, np.arange(self.feed_height)[mask], loc_up[mask]] > weight_up[mask]
            weights[0, np.arange(self.feed_height)[mask], loc_up[mask]] = \
                weight_up[mask]
            warped_image[0, np.arange(self.feed_height)[mask], loc_up[mask]] = \
                image[:, col][mask] / 255.

            mask = loc_down >= 0
            mask[mask] = \
                weights[1, np.arange(self.feed_height)[mask], loc_down[mask]] > weight_down[mask]
            weights[1, np.arange(self.feed_height)[mask], loc_down[mask]] = weight_down[mask]
            warped_image[1, np.arange(self.feed_height)[mask], loc_down[mask]] = \
                image[:, col][mask] / 255.

        weights /= weights.sum(0, keepdims=True) + 1e-7  # normalise
        weights = np.expand_dims(weights, -1)
        warped_image = warped_image[0] * weights[1] + warped_image[1] * weights[0]
        warped_image *= 255.

        # now fill occluded regions with random background
        if not self.disable_background:
            warped_image[warped_image.max(-1) == 0] = background_image[warped_image.max(-1) == 0]

        warped_image = warped_image.astype(np.uint8)

        return warped_image

    def augment_synthetic_image(self, image):

        if self.disable_synthetic_augmentation:

            return Image.fromarray(image.astype(np.uint8))

        # add some noise to stereo image
        noise = np.random.randn(self.feed_height, self.process_width, 3) / 50
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

        do_flip = False
        if self.is_train and random.random() > 0.5:
            do_flip = True

        # load from disk
        left_image, background_image = self.load_images(idx, do_flip=do_flip)
        loaded_disparity = self.load_disparity(idx, do_flip=do_flip)

        inputs['left_image'] = left_image
        inputs['background'] = background_image
        inputs['loaded_disparity'] = loaded_disparity

        # resize and/or crop
        inputs = self.prepare_sizes(inputs)

        # match color in background image
        inputs['background'] = transfer_color(np.array(inputs['background']),
                                              np.array(inputs['left_image']))

        # convert scaleless disparity to pixel disparity
        inputs['disparity'] = \
            self.process_disparity(inputs['loaded_disparity'],
                                   max_disparity_range=(50, self.max_disparity))

        # now generate synthetic stereo image
        projection_disparity = inputs['disparity']
        right_image = self.project_image(inputs['left_image'],
                                         projection_disparity, inputs['background'])

        # augmentation
        right_image = self.augment_synthetic_image(right_image)

        # only keep required keys and prepare for network
        inputs = {'image': inputs['left_image'],
                  'stereo_image': right_image,
                  'disparity': projection_disparity.astype(float),
                  'mono_disparity': inputs['loaded_disparity'].astype(float),
                  }

        # finally crop to feed width
        for key in ['image', 'stereo_image']:
            inputs[key] = inputs[key].crop((0, 0, self.feed_width, self.feed_height))
        for key in ['disparity', 'mono_disparity']:
            inputs[key] = inputs[key][:, :self.feed_width]

        self.preprocess(inputs)
        return inputs
