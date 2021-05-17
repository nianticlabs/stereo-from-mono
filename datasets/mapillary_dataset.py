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


class MapillaryDataset(WarpDataset):

    def __init__(self,
                 data_path,
                 filenames,
                 feed_height,
                 feed_width,
                 max_disparity,
                 is_train=True,
                 disable_normalisation=False,
                 keep_aspect_ratio=True,
                 disable_sharpening=False,
                 monodepth_model='midas',
                 disable_background=False,
                 **kwargs):

        super(MapillaryDataset, self).__init__(data_path, filenames, feed_height, feed_width,
                                               max_disparity,
                                               is_train=is_train, has_gt=True,
                                               disable_normalisation=disable_normalisation,
                                               keep_aspect_ratio=keep_aspect_ratio,
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
        image = self.loader(os.path.join(self.data_path, folder, 'images', frame + '.jpg'))

        if do_flip:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)

        folder, frame = random.choice(self.filenames).split()
        background = self.loader(os.path.join(self.data_path, folder, 'images', frame + '.jpg'))

        # mapillary images are huge -> resize so width is 1200
        w, h = image.size
        new_w = 1200
        new_h = int(h * new_w / w)

        image = image.resize((new_w, new_h), Image.BICUBIC)
        background = background.resize((new_w, new_h), Image.BICUBIC)
        return image, background

    def load_disparity(self, idx, do_flip=False):
        folder, frame = self.filenames[idx].split()
        disparity = np.load(os.path.join(self.data_path, self.disparity_path,
                                         folder, frame + '.npy'))
        disparity = np.squeeze(disparity)

        # mapillary images are huge -> resize so width is 1200
        h, w = disparity.shape
        new_w = 1200
        new_h = int(h * new_w / w)
        # cast disp to float32 for cv2 resizing
        disparity = cv2.resize(disparity.astype(float), (new_w, new_h), interpolation=cv2.INTER_NEAREST)

        if do_flip:
            disparity = disparity[:, ::-1]

        return disparity
