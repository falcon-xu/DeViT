# -*- coding: utf-8 -*-
# @Time    : 2022/4/27 20:53
# @Author  : Falcon
# @FileName: get_dataset.py

import os

from torchvision import datasets, transforms
from torchvision.datasets.folder import ImageFolder

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import create_transform

from .datasets import INatDataset, Flowers102, StanfordCars, OxfordIIITPet


def build_dataset(args):
    transform_train = build_transform(is_train=True, args=args)
    transform_test = build_transform(is_train=False, args=args)

    if args.dataset == 'cifar100':
        train_dataset = datasets.CIFAR100(args.data_path, train=True, transform=transform_train)
        test_dataset = datasets.CIFAR100(args.data_path, train=False, transform=transform_test)
        nb_classes = 100
    elif args.dataset == 'flowers':
        train_dataset = Flowers102(root=args.data_path, split='train', transform=transform_train)
        val_dataset = Flowers102(root=args.data_path, split='val', transform=transform_train)
        train_dataset = train_dataset + val_dataset
        test_dataset = Flowers102(root=args.data_path, split='test', transform=transform_test)
        nb_classes = 102
    elif args.dataset == 'cars':
        train_dataset = StanfordCars(root=args.data_path, split='train', transform=transform_train)
        test_dataset = StanfordCars(root=args.data_path, split='test', transform=transform_test)
        nb_classes = 196
    elif args.dataset == 'pets':
        train_dataset = OxfordIIITPet(root=args.data_path, split='trainval', transform=transform_train)
        test_dataset = OxfordIIITPet(root=args.data_path, split='test', transform=transform_test)
        nb_classes = 37
    elif args.dataset == 'IMNET':
        train_root = os.path.join(args.data_path, 'train')
        test_root = os.path.join(args.data_path, 'val')
        train_dataset = datasets.ImageFolder(train_root, transform=transform_train)
        test_dataset = datasets.ImageFolder(test_root, transform=transform_test)
        nb_classes = 1000
    elif args.dataset == 'INAT':
        train_dataset = INatDataset(args.data_path, train=True, year=2018,
                                    category=args.inat_category, transform=transform_train)
        test_dataset = INatDataset(args.data_path, train=False, year=2018,
                                   category=args.inat_category, transform=transform_test)
        nb_classes = train_dataset.nb_classes
    elif args.dataset == 'INAT19':
        train_dataset = INatDataset(args.data_path, train=True, year=2019,
                                    category=args.inat_category, transform=transform_train)
        test_dataset = INatDataset(args.data_path, train=False, year=2019,
                                   category=args.inat_category, transform=transform_test)
        nb_classes = train_dataset.nb_classes

    return train_dataset, test_dataset, nb_classes


def build_division_dataset(dataset_path, args):
    transform_train = build_transform(is_train=True, args=args)
    transform_test = build_transform(is_train=False, args=args)

    train_dataset = ImageFolder(root=os.path.join(dataset_path, 'train_dataset'), transform=transform_train)
    test_dataset = ImageFolder(root=os.path.join(dataset_path, 'test_dataset'), transform=transform_test)
    num_classes = len(train_dataset.classes)
    return train_dataset, test_dataset, num_classes


def build_transform(is_train, args):
    resize_im = args.input_size > 32
    if is_train:
        if args.no_aug:
            transform = create_transform(
                input_size=args.input_size,
                is_training=True,
            )
        else:

            # this should always dispatch to transforms_imagenet_train
            transform = create_transform(
                input_size=args.input_size,
                is_training=True,
                color_jitter=args.color_jitter,
                auto_augment=args.aa,
                interpolation=args.train_interpolation,
                re_prob=args.reprob,
                re_mode=args.remode,
                re_count=args.recount,
            )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(
                args.input_size, padding=4)
        return transform

    t = []
    if resize_im:
        size = int((256 / 224) * args.input_size)
        t.append(
            transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)
