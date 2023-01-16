import os
import pickle

import PIL.Image
from PIL import Image
import numpy as np
from torchvision.datasets import VisionDataset
from torchvision import transforms, datasets
import torch

from gaussian_blur import GaussianBlur
import matplotlib.pyplot as plt


class CharacterDataset(VisionDataset):
    def __init__(self, dataPath, transform=None, target_transform=None):
        imgsPath = open(dataPath, 'r')
        imgs = []
        for line in imgsPath:
            line = line.rstrip()
            words = line.split()
            imgs.append((words[0], int(words[1])))
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        fn, label = self.imgs[index]
        img = Image.open(fn).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            label = self.transform(label)
        return img, label

    def __len__(self):
        return len(self.imgs)



if __name__ == '__main__':

    def get_simclr_pipeline_transform(size, s=1):
        """Return a set of data augmentation transformations as described in the SimCLR paper."""
        color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
        data_transforms = transforms.Compose([transforms.RandomResizedCrop(size=size),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.RandomVerticalFlip(),
                                              transforms.RandomApply([color_jitter], p=0.8),
                                              transforms.RandomGrayscale(p=0.2),
                                              GaussianBlur(kernel_size=int(0.1 * size)),
                                              transforms.ToTensor()])
        return data_transforms


    import os

    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    trainset =CharacterDataset(dataPath=r'C:\Users\40169\PycharmProjects\similarCharacterDy\SimCLR\datasets\characterImageDocs-train.txt',
                                                                 transform=get_simclr_pipeline_transform(100))
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=8, shuffle=True, num_workers=0, pin_memory=True)
    for step, (tx, ty) in enumerate(trainloader, 0):
        img1 = tx[0].permute(1,2,0).numpy()
        img2 = tx[1].permute(1,2,0).numpy()

        plt.figure()
        plt.subplot(1, 2, 1)
        plt.imshow(img1)
        plt.title("logo")
        plt.subplot(1, 2, 2)
        plt.imshow(img2)
        plt.title("space")
        plt.show()
        break

