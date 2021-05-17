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

from .warp_dataset import WarpDataset

import cv2
cv2.setNumThreads(0)


class MSCOCODataset(WarpDataset):

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

        super(MSCOCODataset, self).__init__(data_path, filenames, feed_height, feed_width,
                                            max_disparity,
                                            is_train=is_train, has_gt=True,
                                            disable_normalisation=disable_normalisation,
                                            keep_aspect_ratio=keep_aspect_ratio,
                                            disable_synthetic_augmentation=
                                            disable_synthetic_augmentation,
                                            disable_sharpening=disable_sharpening,
                                            monodepth_model=monodepth_model,
                                            disable_background=disable_background)

        if self.monodepth_model == 'midas':
            self.disparity_path = 'midas_depths'
        elif self.monodepth_model == 'megadepth':
            self.disparity_path = 'megadepth_depths'
        else:
            raise NotImplementedError

    def load_images(self, idx, do_flip=False):
        """ Load an image to use as left and a random background image to fill in occlusion holes"""

        folder, frame = self.filenames[idx].split()
        image = self.loader(os.path.join(self.data_path, 'images', folder, frame + '.jpg'))

        if do_flip:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)

        folder, frame = random.choice(self.filenames).split()
        background = self.loader(os.path.join(self.data_path, 'images', folder, frame + '.jpg'))

        return image, background

    def load_disparity(self, idx, do_flip=False):
        folder, frame = self.filenames[idx].split()

        disparity = np.load(os.path.join(self.data_path, self.disparity_path,
                                         folder, frame + '.npy'))
        disparity = np.squeeze(disparity)
        if do_flip:
            disparity = disparity[:, ::-1]
        return disparity
