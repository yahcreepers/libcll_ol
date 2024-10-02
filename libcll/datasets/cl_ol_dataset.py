import numpy as np
import copy
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
from PIL import Image


class CLOLDataset(Dataset):
    """
    libcll dataset object

    Parameters
    ----------
    X : Tensor
        the feature of sample set.

    Y : Tensor
        the ground-true labels for corresponding sample.

    num_classes : int
        the number of classes

    Attributes
    ----------
    data : Tensor
        the feature of sample set.

    targets : Tensor
        the complementary labels for corresponding sample.

    true_targets : Tensor
        the ground-truth labels for corresponding sample.

    num_classes : int
        the number of classes.

    input_dim : int
        the feature space after data compressed into a 1D dimension.

    """

    def __init__(
            self, 
            data, 
            targets, 
            true_targets, 
            mask=None, 
            weak_transform=None, 
            strong_transform=None, 
            alg=None, 
    ):
        self.data = data
        self.targets = targets
        self.true_targets = true_targets
        self.mask = mask
        self.weak_transform = weak_transform
        self.strong_transform = strong_transform
        self.alg = alg

    def __getitem__(self, index):
        img, target, true_target = self.data[index], self.targets[index], self.true_targets[index]
        img = Image.fromarray(img)
        if self.mask is not None:
            mask = self.mask[index]
        else:
            mask = torch.tensor(1)
        if self.weak_transform is not None:
            img_w = self.weak_transform(img)
        if self.alg == "ord":
            return img_w, target # uw, ord_labels
        elif self.alg == "freematch":
            return img_w, self.strong_transform(img), target, true_target, mask

    def __len__(self):
        return len(self.data)

    def build_dataset(self, train=True, num_cl=0, transition_matrix=None, noise=None, seed=1126):
        pass