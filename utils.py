import cv2
import numpy as np
import torch
from torch.utils.data import RandomSampler
import yaml


def readlines(filename):
    """ read lines of a text file """
    with open(filename, 'r') as file_handler:
        lines = file_handler.read().splitlines()
    return lines


def normalise_image(img):
    """ Normalize image to [0, 1] range for visualization """
    # img_max = float(img.max().cpu().data)
    # img_min = float(img.min().cpu().data)

    img_max = float(img.max())
    img_min = float(img.min())
    denom = img_max - img_min if img_max != img_min else 1e5
    return (img - img_min) / denom


def transfer_color(target, source):
    target = target.astype(float) / 255
    source = source.astype(float) / 255

    target_means = target.mean(0).mean(0)
    target_stds = target.std(0).std(0)

    source_means = source.mean(0).mean(0)
    source_stds = source.std(0).std(0)

    target -= target_means
    target /= target_stds / source_stds
    target += source_means

    target = np.clip(target, 0, 1)
    target = (target * 255).astype(np.uint8)

    return target


def load_config(config):
    with open(config, 'r') as fh:
        config = yaml.safe_load(fh)
    return config


class MyRandomSampler(RandomSampler):

    def __init__(self, data_source, replacement=False, num_samples=None, start_step=0):
        
        super(MyRandomSampler, self).__init__(data_source,
                                              replacement=replacement,
                                              num_samples=num_samples)

        self.start_step = start_step
        self.to_skip = start_step > 0

    def __iter__(self):
        n = len(self.data_source)
        if self.replacement:
            return iter(torch.randint(high=n, size=(self.num_samples,), dtype=torch.int64).tolist())
        random_idxs = torch.randperm(n).tolist()
        if self.to_skip:
            random_idxs = random_idxs[self.start_step:]
            self.start_step = False
        return iter(random_idxs)
