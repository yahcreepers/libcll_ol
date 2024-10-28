import torch
import pytorch_lightning as pl
import torchvision.transforms as transforms
from torch.utils.data import random_split, Sampler, DataLoader
from torch.utils.data.sampler import BatchSampler, RandomSampler
from torch.utils.data.distributed import DistributedSampler
import numpy as np
import copy
from .cl_ol_dataset import CLOLDataset
from .randaugment import RandAugment
from .utils import get_transition_matrix, collate_fn_multi_label, collate_fn_one_hot

class IndexSampler(Sampler):
    def __init__(self, index):
        self.index = index

    def __iter__(self):
        ind = torch.randperm(len(self.index))
        return iter(self.index[ind].tolist())

    def __len__(self):
        return len(self.index)


class DistributedProxySampler(DistributedSampler):
    """Sampler that restricts data loading to a subset of input sampler indices.

    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSampler instance as a DataLoader sampler,
    and load a subset of the original dataset that is exclusive to it.

    .. note::
        Input sampler is assumed to be of constant size.

    Arguments:
        sampler: Input data sampler.
        num_replicas (optional): Number of processes participating in
            distributed training.
        rank (optional): Rank of the current process within num_replicas.
    """

    def __init__(self, sampler, num_replicas=None, rank=None):        
        super(DistributedProxySampler, self).__init__(sampler, num_replicas=num_replicas, rank=rank, shuffle=False)
        self.sampler = sampler

    def __iter__(self):
        # deterministically shuffle based on epoch
        torch.manual_seed(self.epoch)
        indices = list(self.sampler)

        # add extra samples to make it evenly divisible
        indices += indices[:(self.total_size - len(indices))]
        if len(indices) != self.total_size:
            raise RuntimeError("{} vs {}".format(len(indices), self.total_size))

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        if len(indices) != self.num_samples:
            raise RuntimeError("{} vs {}".format(len(indices), self.num_samples))

        return iter(indices)

class CLDataModule(pl.LightningDataModule):
    def __init__(
        self, 
        dataset_name, 
        dataset_class,
        batch_size=256,
        valid_split=0.1,
        valid_type="URE",
        one_hot=False,
        transition_matrix=None,
        num_cl=1,
        augment=False,
        noise=0.1,
        seed=1126,
        ssl=False, 
        samples_per_class=10, 
        cl_ratio=1.0, 
    ):
        super().__init__()
        self.dataset_name = dataset_name
        self.dataset_class = dataset_class
        self.batch_size = batch_size
        self.valid_split = valid_split
        self.valid_type = valid_type
        self.one_hot = one_hot
        self.transition_matrix = transition_matrix
        self.num_cl = num_cl
        self.augment = augment
        self.noise = noise
        self.seed = seed
        self.ssl = ssl
        self.samples_per_class = samples_per_class
        self.cl_ratio = cl_ratio
    
    def setup(self, stage=None):
        pl.seed_everything(self.seed, workers=True)
        self.train_set = self.dataset_class.build_dataset(self.dataset_name, train=True, num_cl=self.num_cl, transition_matrix=self.transition_matrix, noise=self.noise, seed=self.seed)
        self.test_set = self.dataset_class.build_dataset(self.dataset_name, train=False)
        idx = np.arange(len(self.train_set))
        np.random.shuffle(idx)
        self.train_idx = idx[: int(len(self.train_set) * (1 - self.valid_split))]
        self.valid_idx = idx[int(len(self.train_set) * (1 - self.valid_split)) :]
        
        if self.ssl:
            lb_idx = []
            ulb_idx = []
            if self.dataset_name == "cifar10" and self.samples_per_class == 1:
                lb_idx = [4255, 6446, 8580, 11759, 12598, 29349, 29433, 33759, 35345, 38639]
                ulb_idx = list(set(self.train_idx) - set(lb_idx))
            elif self.dataset_name == "cifar20" and self.samples_per_class == 1:
                lb_idx = [14043,  8148, 34508, 35801, 33949, 30898, 46443, 13186,  5022, 37018, 35314, 26223,  9601, 10361, 40595,  8193, 25226, 47447,  7408, 45044]
                ulb_idx = list(set(self.train_idx) - set(lb_idx))
                # lb_idx = [
                # 7408, 8148, 9850, 10361, 33949, 36506, 37018, 45044, 46443, 47447, 
                # 5022, 8193, 8902, 9601, 25226, 26223, 34089, 35186, 40595, 48024, 
                # 7510, 13186, 14043, 21305, 22805, 31288, 34508, 40470, 41493, 45506, 
                # 9915, 9978, 16631, 19915, 28008, 35314, 35801, 36149, 39215, 42557, 
                # 6695, 14891, 19726, 22715, 23999, 34230, 46511, 47457, 49181, 49397, 
                # 12830, 20293, 26835, 30517, 30898, 31061, 43693, 46501, 47310, 48517, 
                # 1156, 11501, 19974, 21963, 32103, 42189, 46789, 47690, 48229, 48675, 
                # 4255, 6446, 8580, 11759, 12598, 29349, 29433, 33759, 35345, 38639]
                # lb_idx = torch.tensor(lb_idx)
                # lb_i = []
                # for i in range(20):
                #     print(lb_idx[self.train_set.true_targets[lb_idx] == i])
                #     lb_i.extend(lb_idx[self.train_set.true_targets[lb_idx] == i][:2])
                # lb_i = torch.tensor(lb_i)
                # print(lb_i)
                # print(self.train_set.true_targets[lb_i])
                # exit()
            elif self.dataset_name == "cifar20" and self.samples_per_class == 2:
                lb_idx = [
                    14043, 31288,  8148,  9850, 34508, 45506, 35801,  6695, 33949, 41493,
                    30898, 48675, 46443,  7510, 13186, 22715,  5022, 34230, 37018,  8902,
                    35314, 42557, 26223, 48024,  9601, 34089, 10361, 35186, 40595,  9915,
                    8193, 29349, 25226, 22805, 47447, 21305,  7408, 16631, 45044, 19915
                ]
                ulb_idx = list(set(self.train_idx) - set(lb_idx))
            else:
                for c in range(self.train_set.num_classes):
                    c_idx = torch.where(self.train_set.true_targets[self.train_idx] == c)[0]
                    lb_i = np.random.choice(range(c_idx.shape[0]), self.samples_per_class, replace=False)
                    ulb_mask = torch.ones(c_idx.shape[0], dtype=torch.bool)
                    ulb_mask[lb_i] = 0
                    if self.samples_per_class > 1:
                        lb_idx.extend(self.train_idx[c_idx[lb_i]])
                    else:
                        lb_idx.append(self.train_idx[c_idx[lb_i]])
                    ulb_idx.extend(self.train_idx[c_idx[ulb_mask]])
            lb_idx = torch.tensor(lb_idx).long()
            ulb_idx = torch.tensor(ulb_idx).long()
            # print(self.train_set.true_targets[lb_idx])
            lb_data, lb_targets = self.train_set.data[lb_idx], self.train_set.true_targets[lb_idx]
            ulb_data, ulb_targets, ulb_true_targets = self.train_set.data[ulb_idx], torch.cat(self.train_set.targets, dim=0)[ulb_idx], self.train_set.true_targets[ulb_idx]
            cl_mask_index = np.random.choice(range(ulb_data.shape[0]), int(ulb_data.shape[0] * self.cl_ratio), replace=False)
            cl_mask = torch.zeros(ulb_data.shape[0])
            cl_mask[cl_mask_index] = 1
            strong_transform = copy.deepcopy(self.train_set.transform)
            strong_transform.transforms.insert(0, RandAugment(3, 5))
            self.lb_train_set = CLOLDataset(lb_data, lb_targets, lb_targets, weak_transform=self.train_set.transform, alg="ord")
            self.ulb_train_set = CLOLDataset(ulb_data, ulb_targets, ulb_true_targets, mask=cl_mask, weak_transform=self.train_set.transform, strong_transform=strong_transform, alg="freematch")
        
        if self.valid_type == "Accuracy":
            for i in self.valid_idx:
                self.train_set.targets[i] = self.train_set.true_targets[i].view(1)
    
    def train_dataloader(self):
        if self.ssl:
            lb_data_sampler = RandomSampler(self.lb_train_set, True, self.batch_size * torch.cuda.device_count() * self.trainer.max_steps)
            ulb_data_sampler = RandomSampler(self.ulb_train_set, True, self.batch_size * torch.cuda.device_count() * 7 * self.trainer.max_steps)
            if torch.cuda.device_count() > 1:
                lb_data_sampler = DistributedProxySampler(lb_data_sampler)
                ulb_data_sampler = DistributedProxySampler(ulb_data_sampler)
            lb_batch_sampler = BatchSampler(lb_data_sampler, self.batch_size, True)
            ulb_batch_sampler = BatchSampler(ulb_data_sampler, self.batch_size * 7, True)
            lb_train_loader = DataLoader(
                self.lb_train_set, 
                batch_sampler=lb_batch_sampler, 
                num_workers=8,
                persistent_workers=True, 
                pin_memory=True, 
            )
            ulb_train_loader = DataLoader(
                self.ulb_train_set,  
                batch_sampler=ulb_batch_sampler, 
                num_workers=8,
                persistent_workers=True, 
                pin_memory=True, 
            )
            train_loader = {"lb_data": lb_train_loader, "ulb_data": ulb_train_loader}
        else:
            train_sampler = IndexSampler(self.train_idx)
            train_loader = DataLoader(
                self.train_set,
                sampler=train_sampler,
                batch_size=self.batch_size,
                collate_fn=(
                    collate_fn_multi_label
                    if not self.one_hot
                    else lambda batch: collate_fn_one_hot(
                        batch, num_classes=self.train_set.num_classes
                    )
                ),
                shuffle=False,
                num_workers=1, 
                persistent_workers=True, 
                pin_memory=True, 
            )
        return train_loader

    def val_dataloader(self):
        if self.valid_split:
            valid_sampler = IndexSampler(self.valid_idx)
            valid_loader = DataLoader(
                self.train_set,
                sampler=valid_sampler,
                batch_size=self.batch_size,
                collate_fn=collate_fn_multi_label,
                shuffle=False,
                num_workers=8, 
                persistent_workers=True, 
                pin_memory=True, 
            )
        else:
            valid_loader = DataLoader(
                self.test_set, 
                batch_size=self.batch_size, 
                shuffle=False, 
                num_workers=8, 
                persistent_workers=True, 
                pin_memory=True, 
            )
        if self.ssl:
            ulb_valid_loader = DataLoader(
                self.ulb_train_set,  
                batch_size=self.batch_size * 7, 
                num_workers=8, 
                persistent_workers=True, 
                pin_memory=True, 
            )
            valid_loader = [valid_loader, ulb_valid_loader]
        return valid_loader

    def test_dataloader(self):
        test_loader = DataLoader(
            self.test_set, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=8, 
            persistent_workers=True, 
            pin_memory=True, 
        )
        return test_loader
    
    def get_distribution_info(self):
        Q = torch.zeros((self.train_set.num_classes, self.train_set.num_classes))
        for idx in self.train_idx:
            Q[self.train_set.true_targets[idx].long()] += torch.histc(
                self.train_set.targets[idx].float(), self.train_set.num_classes, 0, self.train_set.num_classes
            )
        class_priors = Q.sum(dim=0)
        Q = Q / Q.sum(dim=1).view(-1, 1)
        if self.transition_matrix == "noisy":
            Q = get_transition_matrix("strong", self.train_set.num_classes, self.seed)
        return (
            Q,
            class_priors,
        )