import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision.datasets import CIFAR10
from torchvision import transforms
import torchvision.datasets as dset
import random

def get_loaders(args):
    if args.dataset == "cifar10":
        MEAN = [0.4914, 0.4822, 0.4465]
        STD = [0.2023, 0.1994, 0.2010]
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=MEAN,
                std=STD,
            ),
        ])
        train_dataset = CIFAR10(
            root=args.data,
            train=True,
            download=True,
            transform=train_transform,
        )

        indices = list(range(len(train_dataset)))

        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            sampler=SubsetRandomSampler(indices[:-5000]),
            pin_memory=True,
            num_workers=2,
        )

        reward_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            sampler=SubsetRandomSampler(indices[-5000:]),
            pin_memory=True,
            num_workers=2,
        )

        valid_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=MEAN,
                std=STD,
            ),
        ])
        valid_dataset = CIFAR10(
            root=args.data,
            train=False,
            download=False,
            transform=valid_transform,
        )

        valid_loader = DataLoader(
            valid_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=2,
        )
        # repeat_train_loader = RepeatedDataLoader(train_loader)
        repeat_reward_loader = RepeatedDataLoader(reward_loader)
        repeat_valid_loader = RepeatedDataLoader(valid_loader)

    elif args.dataset == "Sport8" or args.dataset == "MIT67" or args.dataset == "flowers102":
        MEAN = [0.485, 0.456, 0.406]
        STD = [0.229, 0.224, 0.225]
        transf_train = [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(
                brightness=0.4,
                contrast=0.4,
                saturation=0.4,
                hue=0.2)
        ]
        transf_val = [
            transforms.Resize(256),
            transforms.CenterCrop(224),
        ]
        normalize = [
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD)
        ]
        train_transform = transforms.Compose(transf_train + normalize)
        valid_transform = transforms.Compose(transf_val + normalize)
        train_dataset = dset.ImageFolder(root=args.data + "/" + args.dataset + "/train", transform=train_transform)
        valid_dataset = dset.ImageFolder(root=args.data + "/" + args.dataset + "/test", transform=valid_transform)

        n_train = len(train_dataset)
        split = n_train // 2
        indices = list(range(n_train))
        random.shuffle(indices)

        train_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices[:split])
        valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices[split:])

        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            sampler=train_sampler,
            pin_memory=True,
            num_workers=2,
        )

        reward_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            sampler=train_sampler,
            pin_memory=True,
            num_workers=2,
        )

        valid_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            sampler = valid_sampler,
            pin_memory=True,
            num_workers=2,
        )
        # repeat_train_loader = RepeatedDataLoader(train_loader)
        repeat_reward_loader = RepeatedDataLoader(reward_loader)
        repeat_valid_loader = RepeatedDataLoader(valid_loader)


    return train_loader, repeat_reward_loader, repeat_valid_loader


class RepeatedDataLoader():
    def __init__(self, data_loader):
        self.data_loader = data_loader
        self.data_iter = self.data_loader.__iter__()

    def __len__(self):
        return len(self.data_loader)

    def next_batch(self):
        try:
            batch = self.data_iter.__next__()
        except StopIteration:
            self.data_iter = self.data_loader.__iter__()
            batch = self.data_iter.__next__()
        return batch

