import math
import os
import numpy as np
import cv2
from PIL import Image

from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.transforms as transforms
from torchvision import datasets

import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
import numpy as np
import collections
from PIL import Image
import csv


class MimgNetDataset(Dataset):
    def __init__(self, root, mode, resize=84, simclr=False):
        self.simclr = simclr
        if simclr:
            rnd_resizedcrop = transforms.RandomResizedCrop(
                size=resize, scale=(0.08, 1.0), ratio=(0.75, 1.3333333333333333), interpolation=2
            )
            rnd_hflip = transforms.RandomHorizontalFlip(p=0.5)
            color_jitter = transforms.ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2)
            rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
            rnd_gray = transforms.RandomGrayscale(p=0.2)
            normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

            self.transform = transforms.Compose([
                transforms.Resize((resize, resize)),
                rnd_resizedcrop,
                rnd_hflip,
                rnd_color_jitter,
                rnd_gray,
                transforms.ToTensor(),
                normalize
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((resize, resize)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])

        self.path = os.path.join(root, 'images')  # image path
        csvdata = self.loadCSV(os.path.join(root, mode + '.csv'))  # csv path
        self.data = []
        self.img2label = {}
        for i, (k, v) in enumerate(csvdata.items()):
            self.data.append(v)  # [[img1, img2, ...], [img111, ...]]
            self.img2label[k] = i  # {"img_name[:9]":label}

    def loadCSV(self, csvf):
        dictLabels = {}
        with open(csvf) as csvfile:
            csvreader = csv.reader(csvfile, delimiter=',')
            next(csvreader, None)  # skip (filename, label)
            for i, row in enumerate(csvreader):
                filename = row[0]
                label = row[1]
                # append filename to current label
                if label in dictLabels.keys():
                    dictLabels[label].append(filename)
                else:
                    dictLabels[label] = [filename]
        return dictLabels

    def __getitem__(self, index):
        label_ = index // 600
        index_ = index % 600
        pic = Image.open(os.path.join(self.path, self.data[label_][index_])).convert('RGB')
        if self.simclr:
            return self.transform(pic), self.transform(pic.copy())
        else:
            return self.transform(pic)

    def __len__(self):
        # as we have built up to batchsz of sets, you can sample some small batch size of sets.
        return len(self.data) * 600


class CelebADataset(Dataset):
    def __init__(self, root, mode, resize=84, simclr=False):
        self.simclr = simclr
        if not os.path.exists(os.path.join(root, "train.csv")):
            self.generate_CSV(root)
        mean, std = (0.5000, 0.5000, 0.5000), (0.5000, 0.5000, 0.5000)
        if simclr:
            rnd_resizedcrop = transforms.RandomResizedCrop(
                size=resize, scale=(0.2, 1.0),
            )
            rnd_hflip = transforms.RandomHorizontalFlip(p=0.5)
            color_jitter = transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)
            rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
            rnd_gray = transforms.RandomGrayscale(p=0.2)
            normalize = transforms.Normalize(mean, std)

            self.transform = transforms.Compose([
                transforms.Resize((resize, resize)),
                rnd_resizedcrop,
                rnd_hflip,
                rnd_color_jitter,
                rnd_gray,
                transforms.ToTensor(),
                normalize
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((resize, resize)),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])

        self.path = os.path.join(root, 'images')  # image path
        csvdata = self.loadCSV(os.path.join(root, mode + '.csv'))  # csv path
        self.data = []
        self.flatten_data = []
        self.img2label = {}
        for i, (k, v) in enumerate(csvdata.items()):
            self.data.append(v)  # [[img1, img2, ...], [img111, ...]]
            self.img2label[k] = i  # {"img_name[:9]":label}
            self.flatten_data += v


    def generate_CSV(self, root):
        train, val, test = [['filename', 'label']], [["filename", "label"]], [["filename", "label"]]
        with open(os.path.join(root, 'list_eval_partition.txt')) as list_eval_partition:
            with open(os.path.join(root, 'identity_CelebA.txt')) as label_f:
                for line, label in zip(list_eval_partition, label_f):
                    line_data = line.split()
                    label_data = label.split()
                    assert line_data[0] == label_data[0]
                    if line_data[1] == '0':
                        train.append([line_data[0], label_data[1]])
                    elif line_data[1] == '1':
                        val.append([line_data[0], label_data[1]])
                    else:
                        test.append([line_data[0], label_data[1]])
        import csv
        with open(os.path.join(root, 'train.csv'), 'w', encoding='utf-8', newline="") as f:
            csv_writer = csv.writer(f)
            csv_writer.writerows(train)
        with open(os.path.join(root, 'val.csv'), 'w', encoding='utf-8', newline="") as f:
            csv_writer = csv.writer(f)
            csv_writer.writerows(val)
        with open(os.path.join(root, 'test.csv'), 'w', encoding='utf-8', newline="") as f:
            csv_writer = csv.writer(f)
            csv_writer.writerows(test)

    def loadCSV(self, csvf):
        dictLabels = {}
        with open(csvf) as csvfile:
            csvreader = csv.reader(csvfile, delimiter=',')
            next(csvreader, None)  # skip (filename, label)
            for i, row in enumerate(csvreader):
                filename = row[0]
                label = row[1]
                # append filename to current label
                if label in dictLabels.keys():
                    dictLabels[label].append(filename)
                else:
                    dictLabels[label] = [filename]
        return dictLabels

    def __getitem__(self, index):

        pic = Image.open(os.path.join(self.path, self.flatten_data[index])).convert('RGB')
        if self.simclr:
            return self.transform(pic), self.transform(pic.copy())
        else:
            return self.transform(pic)

    def __len__(self):
        # as we have built up to batchsz of sets, you can sample some small batch size of sets.
        return len(self.flatten_data)
