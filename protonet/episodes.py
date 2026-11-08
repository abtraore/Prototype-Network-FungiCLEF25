"""Episode sampling and image transform for the prototypical network.

An episode draws K classes; for each class S support images and Q=1 query
image. Classes with fewer than S images are sampled with replacement, and when
no image is left for the query it is drawn from the class again (most species
have only one to four images).
"""

import random

import albumentations as A
import cv2
import numpy as np
import torch

from fungiclef.data import CLIP_MEAN, CLIP_STD, read_rgb


def make_transform(size):
    tfm = A.Compose(
        [
            A.Resize(height=size, width=size, interpolation=cv2.INTER_CUBIC),
            A.D4(),  # random flips / 90-degree rotations
            A.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
        ]
    )

    def apply(path):
        x = tfm(image=read_rgb(path))["image"]
        return torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))

    return apply


class EpisodeSampler:
    def __init__(self, df, transform, k, s):
        self.paths = df.groupby("category_id")["path"].apply(list).to_dict()
        self.classes = list(self.paths)
        self.transform, self.k, self.s = transform, k, s

    def sample(self):
        classes = random.sample(self.classes, self.k)
        support, query = [], []
        for c in classes:
            paths = self.paths[c]
            if len(paths) >= self.s:
                sup = random.sample(paths, self.s)
                rest = [p for p in paths if p not in sup]
                q = random.choice(rest or paths)
            else:
                sup = random.choices(paths, k=self.s)
                q = random.choice(paths)
            support.append(torch.stack([self.transform(p) for p in sup]))
            query.append(self.transform(q))
        order = torch.randperm(self.k)  # shuffle queries; target = class index
        return torch.stack(support), torch.stack(query)[order], order
