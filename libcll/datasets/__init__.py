import torch
import torchvision.transforms as transforms
from torch.utils.data import random_split, Sampler, DataLoader
from torch.utils.data.sampler import BatchSampler, RandomSampler
import numpy as np
import copy
from .cl_base_dataset import CLBaseDataset
from .cl_cifar10 import CLCIFAR10
from .cl_cifar20 import CLCIFAR20
from .cl_yeast import CLYeast
from .cl_texture import CLTexture
from .cl_control import CLControl
from .cl_dermatology import CLDermatology
from .cl_fmnist import CLFMNIST
from .cl_kmnist import CLKMNIST
from .cl_mnist import CLMNIST
from .cl_micro_imagenet10 import CLMicro_ImageNet10
from .cl_micro_imagenet20 import CLMicro_ImageNet20
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


def prepare_dataloader(
    dataset,
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
):

    if dataset == "mnist":
        train_set = CLMNIST(
            root="./data/mnist",
            train=True,
        )
        test_set = CLMNIST(root="./data/mnist", train=False)

        Q = get_transition_matrix(transition_matrix, train_set.num_classes, noise, seed)
        train_set.gen_complementary_target(num_cl, Q)

    elif dataset == "fmnist":
        train_set = CLFMNIST(
            root="./data/fmnist",
            train=True,
        )
        test_set = CLFMNIST(root="./data/fmnist", train=False)

        Q = get_transition_matrix(transition_matrix, train_set.num_classes, noise, seed)
        train_set.gen_complementary_target(num_cl, Q)

    elif dataset == "kmnist":
        train_set = CLKMNIST(
            root="./data/kmnist",
            train=True,
        )
        test_set = CLKMNIST(root="./data/kmnist", train=False)

        Q = get_transition_matrix(transition_matrix, train_set.num_classes, noise, seed)
        train_set.gen_complementary_target(num_cl, Q)

    elif dataset == "cifar10":
        if augment:
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
        else:
            train_transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(
                        [0.4914, 0.4822, 0.4465], [0.247, 0.2435, 0.2616]
                    ),
                ]
            )
        test_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.2435, 0.2616]),
            ]
        )
        train_set = CLCIFAR10(
            root="./data/cifar10",
            train=True,
            transform=train_transform,
        )
        test_set = CLCIFAR10(
            root="./data/cifar10",
            train=False,
            transform=test_transform,
        )

        Q = get_transition_matrix(transition_matrix, train_set.num_classes, noise, seed)
        train_set.gen_complementary_target(num_cl, Q)

    elif dataset == "cifar20":
        if augment:
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
        else:
            train_transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(
                        [0.4914, 0.4822, 0.4465], [0.247, 0.2435, 0.2616]
                    ),
                ]
            )
        test_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.2435, 0.2616]),
            ]
        )
        train_set = CLCIFAR20(
            root="./data/cifar20",
            train=True,
            transform=train_transform,
        )
        test_set = CLCIFAR20(
            root="./data/cifar20",
            train=False,
            transform=test_transform,
        )

        Q = get_transition_matrix(transition_matrix, train_set.num_classes, noise, seed)
        train_set.gen_complementary_target(num_cl, Q)

    elif dataset == "yeast":
        train_set = CLYeast(
            root="./data/yeast",
            train=True,
        )
        test_set = CLYeast(
            root="./data/yeast",
            train=False,
        )

        Q = get_transition_matrix(transition_matrix, train_set.num_classes, noise, seed)
        train_set.gen_complementary_target(num_cl, Q)

    elif dataset == "texture":
        train_set = CLTexture(
            root="./data/texture",
            train=True,
        )
        test_set = CLTexture(
            root="./data/texture",
            train=False,
        )

        Q = get_transition_matrix(transition_matrix, train_set.num_classes, noise, seed)
        train_set.gen_complementary_target(num_cl, Q)

    elif dataset == "dermatology":
        train_set = CLDermatology(
            root="./data/dermatology",
            train=True,
        )
        test_set = CLDermatology(
            root="./data/dermatology",
            train=False,
        )

        Q = get_transition_matrix(transition_matrix, train_set.num_classes, noise, seed)
        train_set.gen_complementary_target(num_cl, Q)

    elif dataset == "control":
        train_set = CLControl(
            root="./data/control",
            train=True,
        )
        test_set = CLControl(
            root="./data/control",
            train=False,
        )

        Q = get_transition_matrix(transition_matrix, train_set.num_classes, noise, seed)
        train_set.gen_complementary_target(num_cl, Q)

    elif dataset == "clcifar10":
        if augment:
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
        else:
            train_transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(
                        [0.4914, 0.4822, 0.4465], [0.247, 0.2435, 0.2616]
                    ),
                ]
            )
        test_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.2435, 0.2616]),
            ]
        )
        train_set = CLCIFAR10(
            root="./data/cifar10",
            train=True,
            transform=train_transform,
            num_cl=num_cl,
        )
        test_set = CLCIFAR10(
            root="./data/cifar10",
            train=False,
            transform=test_transform,
            num_cl=num_cl,
        )

    elif dataset == "clcifar20":
        if augment:
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
        else:
            train_transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(
                        [0.4914, 0.4822, 0.4465], [0.247, 0.2435, 0.2616]
                    ),
                ]
            )
        test_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.2435, 0.2616]),
            ]
        )
        train_set = CLCIFAR20(
            root="./data/cifar20",
            train=True,
            transform=train_transform,
            num_cl=num_cl,
        )
        test_set = CLCIFAR20(
            root="./data/cifar20",
            train=False,
            transform=test_transform,
            num_cl=num_cl,
        )

    elif dataset == "micro_imagenet10":
        train_transform = transforms.Compose(
            [
                transforms.RandomCrop(64, padding=8),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )
        test_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )
        train_set = CLMicro_ImageNet10(
            root="./data/imagenet10",
            train=True,
            transform=train_transform,
            num_cl=num_cl,
        )
        test_set = CLMicro_ImageNet10(
            root="./data/imagenet10",
            train=False,
            transform=test_transform,
            num_cl=num_cl,
        )
        Q = get_transition_matrix(transition_matrix, train_set.num_classes, noise, seed)
        train_set.gen_complementary_target(num_cl, Q)

    elif dataset == "micro_imagenet20":
        train_transform = transforms.Compose(
            [
                transforms.RandomCrop(64, padding=8),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )
        test_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )
        train_set = CLMicro_ImageNet20(
            root="./data/imagenet20",
            train=True,
            transform=train_transform,
            num_cl=num_cl,
        )
        test_set = CLMicro_ImageNet20(
            root="./data/imagenet20",
            train=False,
            transform=test_transform,
            num_cl=num_cl,
        )
        Q = get_transition_matrix(transition_matrix, train_set.num_classes, noise, seed)
        train_set.gen_complementary_target(num_cl, Q)

    elif dataset == "clmicro_imagenet10":
        train_transform = transforms.Compose(
            [
                transforms.RandomCrop(64, padding=8),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )
        test_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )
        train_set = CLMicro_ImageNet10(
            root="./data/imagenet10",
            train=True,
            transform=train_transform,
            num_cl=num_cl,
        )
        test_set = CLMicro_ImageNet10(
            root="./data/imagenet10",
            train=False,
            transform=test_transform,
            num_cl=num_cl,
        )

    elif dataset == "clmicro_imagenet20":
        train_transform = transforms.Compose(
            [
                transforms.RandomCrop(64, padding=8),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )
        test_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )
        train_set = CLMicro_ImageNet20(
            root="./data/imagenet20",
            train=True,
            transform=train_transform,
            num_cl=num_cl,
        )
        test_set = CLMicro_ImageNet20(
            root="./data/imagenet20",
            train=False,
            transform=test_transform,
            num_cl=num_cl,
        )

    else:
        raise NotImplementedError

    idx = np.arange(len(train_set))
    np.random.shuffle(idx)
    train_idx = idx[: int(len(train_set) * (1 - valid_split))]
    valid_idx = idx[int(len(train_set) * (1 - valid_split)) :]
    train_sampler = IndexSampler(train_idx)
    valid_sampler = IndexSampler(valid_idx)
    if valid_type == "Accuracy":
        for i in valid_idx:
            train_set.targets[i] = train_set.true_targets[i].view(1)
    
    if ssl:
        lb_idx = []
        ulb_idx = []
        for c in range(train_set.num_classes):
            c_idx = torch.where(train_set.true_targets[train_idx] == c)[0]
            lb_i = np.random.choice(range(c_idx.shape[0]), samples_per_class, replace=False)
            ulb_mask = torch.ones(c_idx.shape[0], dtype=torch.bool)
            ulb_mask[lb_i] = 0
            lb_idx.extend(train_idx[c_idx[lb_i]])
            ulb_idx.extend(train_idx[c_idx[ulb_mask]])
        lb_idx = torch.tensor(lb_idx)
        ulb_idx = torch.tensor(ulb_idx)
        # print("lb:", lb_idx.shape, torch.cat((torch.tensor(train_idx), lb_idx)).unique().shape)
        # print("ulb:", ulb_idx.shape, torch.cat((torch.tensor(train_idx), ulb_idx)).unique().shape)
        lb_data, lb_targets = train_set.data[lb_idx], train_set.true_targets[lb_idx]
        ulb_data, ulb_targets, ulb_true_targets = train_set.data[ulb_idx], torch.tensor(train_set.targets)[ulb_idx], train_set.true_targets[ulb_idx]
        strong_transform = copy.deepcopy(train_set.transform)
        strong_transform.transforms.insert(0, RandAugment(3, 5))
        lb_train_set = CLOLDataset(lb_data, lb_targets, lb_targets, weak_transform=train_set.transform, alg="ord")
        ulb_train_set = CLOLDataset(ulb_data, ulb_targets, ulb_true_targets, weak_transform=train_set.transform, strong_transform=strong_transform, alg="freematch")
        lb_data_sampler = RandomSampler(lb_train_set, True, batch_size * 2 ** 20)
        lb_batch_sampler = BatchSampler(lb_data_sampler, batch_size, True)
        ulb_data_sampler = RandomSampler(ulb_train_set, True, batch_size * 7 * 2 ** 20)
        ulb_batch_sampler = BatchSampler(ulb_data_sampler, batch_size * 7, True)
        # print(batch_size)
        def collate_fn(batch):
            # print("WWW", batch)
            # print("AAA", len(batch))
            return batch
        lb_train_loader = DataLoader(
            lb_train_set, 
            batch_sampler=lb_batch_sampler, 
            # sampler=data_sampler, 
            # batch_size=batch_size, 
            # shuffle=True,
            # collate_fn=collate_fn, 
            num_workers=4,
        )
        ulb_train_loader = DataLoader(
            ulb_train_set,  
            batch_sampler=ulb_batch_sampler, 
            # batch_size=batch_size * 7, 
            # shuffle=True,
            num_workers=4,
        )
        train_loader = {"lb_data": lb_train_loader, "ulb_data": ulb_train_loader}
    else:
        train_loader = DataLoader(
            train_set,
            sampler=train_sampler,
            batch_size=batch_size,
            collate_fn=(
                collate_fn_multi_label
                if not one_hot
                else lambda batch: collate_fn_one_hot(
                    batch, num_classes=train_set.num_classes
                )
            ),
            shuffle=False,
            num_workers=4,
        )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=4
    )
    if valid_split:
        valid_loader = DataLoader(
            train_set,
            sampler=valid_sampler,
            batch_size=batch_size,
            collate_fn=collate_fn_multi_label,
            shuffle=False,
            num_workers=4,
        )
    else:
        valid_loader = test_loader
    if ssl:
        ulb_valid_loader = DataLoader(
            ulb_train_set,  
            batch_size=batch_size * 7, 
            num_workers=4,
        )
        valid_loader = [valid_loader, ulb_valid_loader]
    Q = torch.zeros((train_set.num_classes, train_set.num_classes))
    for idx in train_idx:
        Q[train_set.true_targets[idx].long()] += torch.histc(
            train_set.targets[idx].float(), train_set.num_classes, 0, train_set.num_classes
        )
    class_priors = Q.sum(dim=0)
    Q = Q / Q.sum(dim=1).view(-1, 1)
    if transition_matrix == "noisy":
        Q = get_transition_matrix("strong", train_set.num_classes, seed)
    return (
        train_loader,
        valid_loader,
        test_loader,
        train_set.input_dim,
        train_set.num_classes,
        Q,
        class_priors,
    )
