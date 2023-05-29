# -*- coding: utf-8 -*-
# @Time    : 2022/4/29 14:27
# @Author  : Falcon
# @FileName: splite_dataset.py
import os.path
import argparse
from data.datasets import INatDataset, Flowers102, StanfordCars, OxfordIIITPet
from torchvision.datasets.folder import ImageFolder
from torchvision import datasets
import random
import shutil
from tqdm import tqdm


def get_args_parser():
    parser = argparse.ArgumentParser('ViT training and evaluation script', add_help=False)
    parser.add_argument('--data_path', default=r'./datasets', type=str)
    parser.add_argument('--dataset', default='cifar100',
                        choices=['cifar100', 'flowers', 'cars', 'pets', 'IMNET', 'INAT', 'INAT19'],
                        type=str)
    parser.add_argument('--num_division', default=6, type=int)
    parser.add_argument('--output_path',
                        default=r'./output',
                        type=str)

    return parser


def main(args):
    num_sub = args.num_division
    path = args.data_path
    args.output_path = os.path.join(args.output_path, f'division{args.num_division}')

    if args.dataset == 'flower':

        for mode in ['train', 'test']:

            if mode == 'train':
                dataset = Flowers102(root=path, split='train')
                dataset1 = Flowers102(root=path, split='val')

                num_classes = dataset.num_classes
                image_files = dataset._image_files + dataset1._image_files
                labels = dataset._labels + dataset1._labels
            else:
                dataset = Flowers102(root=path, split='test')
                num_classes = dataset.num_classes
                image_files = dataset._image_files
                labels = dataset._labels

            random.seed(42)
            label_list = list(range(num_classes))
            random.shuffle(label_list)

            splite_lable_lists = [label_list[i * num_classes // num_sub:(i + 1) * num_classes // num_sub]
                                  if i < num_sub - 1 else label_list[i * num_classes // num_sub:] for i in range(num_sub)]

            for i, file in enumerate(tqdm(image_files)):
                for sub_index in range(num_sub):
                    if labels[i] in splite_lable_lists[sub_index]:
                        output_file_path = os.path.join(args.output_path, args.dataset, f'sub-dataset{sub_index}',
                                                        f'{mode}_dataset',
                                                        str(labels[i]))
                        os.makedirs(output_file_path, exist_ok=True)
                        output_file_path = os.path.join(output_file_path, file.name)
                        shutil.copy(file, output_file_path)

    elif args.dataset == 'car':
        for mode in ['train', 'test']:
            dataset = StanfordCars(root=path, split=mode)
            num_classes = dataset.num_classes
            image_files = []
            labels = []
            for sample in dataset._samples:
                image_files.append(sample[0])
                labels.append(sample[1])

            random.seed(42)
            label_list = list(range(num_classes))
            random.shuffle(label_list)

            splite_lable_lists = [label_list[i * num_classes // num_sub:(i + 1) * num_classes // num_sub]
                                  if i < num_sub - 1 else label_list[i * num_classes // num_sub:] for i in range(num_sub)]

            for i, file in enumerate(tqdm(image_files)):
                for sub_index in range(num_sub):
                    if labels[i] in splite_lable_lists[sub_index]:
                        output_file_path = os.path.join(args.output_path, args.dataset, f'sub-dataset{sub_index}',
                                                        f'{mode}_dataset',
                                                        str(labels[i]))
                        os.makedirs(output_file_path, exist_ok=True)
                        output_file_path = os.path.join(output_file_path, file.split('\\')[-1])
                        shutil.copy(file, output_file_path)

    elif args.dataset == 'pet':
        for mode in ['train', 'test']:
            dataset = OxfordIIITPet(root=path, split='trainval') if mode == 'train' else OxfordIIITPet(root=path,
                                                                                                       split='test')
            num_classes = dataset.num_classes
            image_files = dataset._images
            labels = dataset._labels

            random.seed(42)
            label_list = list(range(num_classes))
            random.shuffle(label_list)

            splite_lable_lists = [label_list[i * num_classes // num_sub:(i + 1) * num_classes // num_sub]
                                  if i < num_sub - 1 else label_list[i * num_classes // num_sub:] for i in range(num_sub)]

            for i, file in enumerate(tqdm(image_files)):
                for sub_index in range(num_sub):
                    if labels[i] in splite_lable_lists[sub_index]:
                        output_file_path = os.path.join(args.output_path, args.dataset, f'sub-dataset{sub_index}',
                                                        f'{mode}_dataset',
                                                        str(labels[i]))
                        os.makedirs(output_file_path, exist_ok=True)
                        output_file_path = os.path.join(output_file_path, file.name)
                        shutil.copy(file, output_file_path)

    elif args.dataset == 'IMNET':
        train_root = os.path.join(args.data_path, 'train')
        val_root = os.path.join(args.data_path, 'val')
        train_dataset = ImageFolder(train_root)
        val_dataset = ImageFolder(val_root)

        classes = train_dataset.classes
        num_classes = len(classes)
        class2idx = train_dataset.class_to_idx
        idx2class = {i:c for i,c in enumerate(class2idx.keys())}

        random.seed(42)
        label_list = list(range(num_classes))
        random.shuffle(label_list)

        splite_lable_lists = [label_list[i * num_classes // num_sub:(i + 1) * num_classes // num_sub]
                              if i < num_sub - 1 else label_list[i * num_classes // num_sub:] for i in range(num_sub)]

        for i, divisions in enumerate(splite_lable_lists):
            print(f'Start to create NO.{i} sub-dataset.')
            for sub_idx in tqdm(divisions):
                label = idx2class[sub_idx]
                train_output_path = os.path.join(args.output_path, f'sub-dataset{i}', 'train_dataset', label)
                val_output_path = os.path.join(args.output_path, f'sub-dataset{i}', 'test_dataset', label)
                train_imgs_path = os.path.join(train_root, label)
                val_imgs_path = os.path.join(val_root, label)
                shutil.copytree(train_imgs_path, train_output_path)
                shutil.copytree(val_imgs_path, val_output_path)

    elif args.dataset == 'cifar100':
        train_root = os.path.join(args.data_path, 'train')
        val_root = os.path.join(args.data_path, 'val')
        train_dataset = ImageFolder(train_root)
        val_dataset = ImageFolder(val_root)

        classes = train_dataset.classes
        num_classes = len(classes)
        class2idx = train_dataset.class_to_idx
        idx2class = {i: c for i, c in enumerate(class2idx.keys())}

        random.seed(42)
        label_list = list(range(num_classes))
        random.shuffle(label_list)

        splite_lable_lists = [label_list[i * num_classes // num_sub:(i + 1) * num_classes // num_sub]
                              if i < num_sub - 1 else label_list[i * num_classes // num_sub:] for i in range(num_sub)]

        for i, divisions in enumerate(splite_lable_lists):
            print(f'Start to create NO.{i} sub-dataset.')
            for sub_idx in tqdm(divisions):
                label = idx2class[sub_idx]
                train_output_path = os.path.join(args.output_path, f'sub-dataset{i}', 'train_dataset', label)
                val_output_path = os.path.join(args.output_path, f'sub-dataset{i}', 'test_dataset', label)
                train_imgs_path = os.path.join(train_root, label)
                val_imgs_path = os.path.join(val_root, label)
                shutil.copytree(train_imgs_path, train_output_path)
                shutil.copytree(val_imgs_path, val_output_path)



if __name__ == '__main__':
    parser = argparse.ArgumentParser('dataset partition', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
