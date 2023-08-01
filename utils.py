import torch
import numpy as np
import json
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
import pandas as pd
from PIL import Image
import os
from glob import glob


class ImageFolderWithPaths(ImageFolder):
    def __getitem__(self, index):
        img, label = super(ImageFolderWithPaths, self).__getitem__(index)

        path = self.imgs[index][0]

        return (img, label, path)
        

class ImageFolderByPath(Dataset):
    def __init__(self, root, label_info, transform):
        df = pd.read_csv(label_info)
        df = np.array(df[["ImageId", "TrueLabel"]])
        datas = [(root +imageid, label) for imageid, label in df]
        self.datas = datas
        self.transform = transform

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, index):
        img_path, label = self.datas[index]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label, img_path


class ImageFolderTransfer(Dataset):
    """
    root: adv_path
    """
    def __init__(self, root, transform):
        self.datas = glob(os.path.join(root, "*.JPEG"))
        if len(self.datas) ==0:
            self.datas = glob(os.path.join(root, "*.png"))
        if len(self.datas) ==0:
            self.datas = glob(os.path.join(root, "*.jpg"))
        self.transform = transform

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, index):
        img_path = self.datas[index]
        clean_img_path = img_path.replace("/adv", "/clean")
        image = Image.open(img_path).convert("RGB")
        clean_image = Image.open(clean_img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)
            clean_image = self.transform(clean_image)

        return image, clean_image

class Normalize(torch.nn.Module):
    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        if not isinstance(mean, torch.Tensor):
            mean = torch.tensor(mean)
        if not isinstance(std, torch.Tensor):
            std = torch.tensor(std)
        self.register_buffer('mean', mean)
        self.register_buffer('std', std)

    def forward(self, tensor):
        return normalize_fn(tensor, self.mean, self.std)

    def backward(self):
        return "mean={}, std={}".format(self.mean, self.std)

def normalize_fn(tensor, mean, std):
    """
    Differentiable version of torchvision.functional.normalize
    - default assumes color channel is at dim = 1
    """
    mean = mean[None, :, None, None]
    std = std[None, :, None, None]
    return tensor.sub(mean).div(std)


def get_imagenet_dicts():
    idx2label = []
    cls2label = {}
    with open("imagenet_class_index.json", "r") as read_file:
        class_idx = json.load(read_file)
        idx2label = [class_idx[str(k)][1] for k in range(len(class_idx))]
        cls2label = {class_idx[str(k)][0]: class_idx[str(k)][1] for k in range(len(class_idx))}

    return idx2label, cls2label

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)