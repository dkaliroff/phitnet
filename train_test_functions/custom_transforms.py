from __future__ import division
import numpy as np
import random
from PIL import Image
import torchvision.transforms.functional as TVF

'''Set of tranform random routines that takes list of inputs as arguments,
in order to have random but coherent transformations.'''

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def append(self, transform):
        self.transforms.append(transform)

    def __call__(self, images):
        for t in self.transforms:
            images = t(images)
        return images

class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std


    def __call__(self, images):
        for idx, tensor in enumerate(images):
            for t, m, s in zip(tensor, self.mean, self.std):
                t.sub_(m).div_(s)
        return images

class ArrayToTensor(object):
    """Converts a list of numpy.ndarray (H x W x C) along with a intrinsics matrix to a list of torch.FloatTensor of shape (C x H x W) with a intrinsics tensor."""

    def __call__(self, images):
        tensors = []
        for im in images:
            tensors.append(TVF.to_tensor(im))
        return tensors

class Grayscale(object):
    """Converts a list of numpy.ndarray (H x W x C) along with a intrinsics matrix to a list of torch.FloatTensor of shape (C x H x W) with a intrinsics tensor."""

    def __call__(self, images):
        return [TVF.to_grayscale(im,1) for im in images]

class RandomHorizontalFlip(object):
    """Randomly horizontally flips the given numpy array with a probability of 0.5"""

    def __call__(self, images):
        if random.random() < 0.5:
            output_images = [im.transpose(Image.FLIP_LEFT_RIGHT) for im in images]
        else:
            output_images = images
        return output_images

class RandomScaleCrop(object):
    """Randomly zooms images up to 15% and crop them to keep same size as before."""

    def __call__(self, images):

        in_w,in_h = images[0].size
        x_scaling = np.random.uniform(1,1.1,1)
        y_scaling = x_scaling
        scaled_h, scaled_w = int(in_h * y_scaling), int(in_w * x_scaling)

        for ii in range(len(images)):
            images[ii]=images[ii].resize((scaled_w, scaled_h), Image.BILINEAR)

        offset_y = np.random.randint(scaled_h - in_h + 1)
        offset_x = np.random.randint(scaled_w - in_w + 1)

        for ii in range(len(images)):
            images[ii]=images[ii].crop((offset_x, offset_y, in_w+offset_x, in_h+offset_y))

        return images
