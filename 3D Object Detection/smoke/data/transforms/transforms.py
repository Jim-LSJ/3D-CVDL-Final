import torch
from torchvision.transforms import functional as F


class Compose():
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target, depth):
        for t in self.transforms:
            image, target, depth = t(image, target, depth)
        return image, target, depth


class ToTensor():
    def __call__(self, image, target, depth):
        if depth is not None:
            return F.to_tensor(image), target, F.to_tensor(depth)
        else:
            return F.to_tensor(image), target, depth


class Normalize():
    def __init__(self, mean, std, to_bgr=True):
        self.mean = mean
        self.std = std
        self.to_bgr = to_bgr

    def __call__(self, image, target, depth):
        if self.to_bgr:
            image = image[[2, 1, 0]]
        image = F.normalize(image, mean=self.mean, std=self.std)
        depth = F.normalize(depth, mean=[0.5], std=[0.5]) if depth is not None else depth
        return image, target, depth
