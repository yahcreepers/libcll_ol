import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import urllib.request
from tqdm import tqdm
import pickle
import gdown
import os
from libcll.datasets.cl_base_dataset import CLBaseDataset
from libcll.datasets.utils import get_transition_matrix


class CLCIFAR100(torchvision.datasets.CIFAR100, CLBaseDataset):
    """

    Real-world complementary-label dataset. Call ``gen_complementary_target()`` if you want to access synthetic complementary labels.

    Parameters
    ----------
    root : str
        path to store dataset file.

    train : bool
        training set if True, else testing set.

    transform : callable, optional
        a function/transform that takes in a PIL image and returns a transformed version.

    target_transform : callable, optional
        a function/transform that takes in the target and transforms it.

    download : bool
        if true, downloads the dataset from the internet and puts it in root directory. If dataset is already downloaded, it is not downloaded again.

    num_cl : int
        the number of real-world complementary labels of each data chosen from [1, 3].

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
        root="./data/cifar100",
        train=True,
        transform=None,
        target_transform=None,
        download=True,
    ):
        super(CLCIFAR100, self).__init__(
            root, train, transform, target_transform, download
        )
        self.targets = torch.Tensor(self.targets)
        self.num_classes = 100
        self.input_dim = 3 * 32 * 32
    
    @classmethod
    def build_dataset(self, dataset_name=None, train=True, num_cl=0, transition_matrix=None, noise=None, seed=1126):
        if train:
            train_transform = transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomCrop(32, padding=4),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        [0.4914, 0.4822, 0.4465], [0.247, 0.2435, 0.2616]
                    ),
                ]
            )
            dataset = self(
                train=True,
                transform=train_transform,
            )
            Q = get_transition_matrix(transition_matrix, dataset.num_classes, noise, seed)
            dataset.gen_complementary_target(num_cl, Q)
        else:
            test_transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.2435, 0.2616]),
                ]
            )
            dataset = self(
                train=False,
                transform=test_transform,
            )
        return dataset