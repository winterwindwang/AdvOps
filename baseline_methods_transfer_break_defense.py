import argparse
import os
import pickle

import numpy as np
import torch
from matplotlib import pyplot as plt

from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from utils import *
from generator import OpsAdvGenerator, Discriminator
import collections
from collections import Counter
import json
import copy
import timm
from io import BytesIO
import torchvision
from random import randint, uniform
from torch_denoise_tv_chambol import denoise_tv_chambolle_torch
from glob import glob

def defult_fn(img_path):
    return Image.open(img_path).convert("RGB")

class TestDefenseDataset(Dataset):
    def __init__(self, data_dir, transform=None, defult_fn=defult_fn) -> None:
        super().__init__()
        self.datas = np.load(data_dir)
        self.transform = transform
        self.defult_fn = defult_fn
    
    def __len__(self):
        return len(self.datas)

    def __getitem__(self, index):
        image_path, label = self.datas[index]
        img = self.defult_fn(str(image_path))
        if self.transform is not None:
            img = self.transform(img)
        return img, int(label)


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def randomJPEGcompression(image, qf=75):
    """https://discuss.pytorch.org/t/how-can-i-develop-a-transformation-that-performs-jpeg-compression-with-a-random-qf/43588/5"""
    # outputIoStream = BytesIO()
    # image.save(outputIoStream, "JPEG", quality=qf, optimice=True)
    # outputIoStream.seek(0)
    # image = outputIoStream.read()
    image_res = None
    with BytesIO() as output:
        image.save(output, "JPEG", quality=qf, optimice=True)
        output.seek(0)
        image_jpeg = Image.open(output)
        image_res = copy.deepcopy(image_jpeg)
    return image_res

def jpeg_defense(images, jpeg_transformation):
    """JPEG compression corruption"""
    images_pl = [torchvision.transforms.ToPILImage()(img) for img in images]
    image_jpeg = torch.stack(list(map(jpeg_transformation, images_pl)), dim=0)
    return image_jpeg


def baussian_blur_defense(images, gaussian_blur_transformation):
    """Gaussian blur corruption"""
    images_pl = [torchvision.transforms.ToPILImage()(img) for img in images]
    image_blur = torch.stack(list(map(gaussian_blur_transformation, images_pl)), dim=0)
    return image_blur


def pixel_deflection_without_map(imges, deflections=200, window=10):
    """
    paper: Deflecting Adversarial Attacks with Pixel Deflection
    :param imges:
    :param deflections:
    :param window:
    :return:
    """
    img = imges
    N,C,H, W = img.size()
    while deflections > 0:
        #for consistency, when we deflect the given pixel from all the three channels.
        for c in range(C):
            x,y = randint(0,H-1), randint(0,W-1)
            while True: #this is to ensure that PD pixel lies inside the image
                a,b = randint(-1*window,window), randint(-1*window,window)
                if x+a < H and x+a > 0 and y+b < W and y+b > 0: break
            # calling pixel deflection as pixel swap would be a misnomer,
            # as we can see below, it is one way copy
            img[:,c, x,y] = img[:,c, x+a,y+b]
        deflections -= 1
    return img


def test_defense_accuracy(model, dataloader, defense_method, device):
    acc_cnt = 0
    fr_cnt = 0
    total = 0
    for i, batch in enumerate(dataloader):

        images, labels = batch
        images, labels = images.to(device), labels.to(device)


        if "tvm" in defense_method:
            images = denoise_tv_chambolle_torch(images, multichannel=True)
        elif "pixel" in defense_method:
            images = pixel_deflection_without_map(images)
        
        output = model(images)
        pred = torch.argmax(output, dim=1)

        acc_cnt += (pred == labels).sum().item()
        fr_cnt += (pred != labels).sum().item()

        total += images.size(0)

    fr = round(100*(fr_cnt)/ total, 2)
    acc = round(100*(acc_cnt)/ total, 2)
    return acc, fr, total


def get_model_list(model_name):
    normalize = Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225])
    if "resnet50" in model_name.lower():
        model = models.resnet50(pretrained=True)
    elif "vgg16" in model_name.lower():
        model = models.vgg16(pretrained=True)
    elif "densenet121" in model_name.lower():
        model = models.densenet121(pretrained=True)
    elif "resnext" in model_name.lower():
        model = models.resnext50_32x4d(pretrained=True)
    elif "wideresnet" in model_name.lower():
        model = models.wide_resnet50_2(pretrained=True)
    elif "mnasnet" in model_name.lower():
        model = models.mnasnet1_0(pretrained=True)
    elif "squeezenet" in model_name.lower():
        model = models.squeezenet1_0(pretrained=True)
    elif "mlp" in model_name:
        model = timm.create_model('mixer_b16_224', pretrained=True)
    elif "adv_inception_v3" in model_name:
        model = timm.create_model('adv_inception_v3', pretrained=True)
    elif "env_adv_incep_res_v2" in model_name:
        model = timm.create_model('ens_adv_inception_resnet_v2', pretrained=True) #, num_classes=0, global_pool='')
    elif "adv_incep_v2" in model_name:
        model = timm.create_model("inception_resnet_v2", pretrained=True)
    model = torch.nn.Sequential(normalize, model)
    model.eval()
    model.to(device)
    return model

def get_args():
    parser = argparse.ArgumentParser(description="Args Container")
    parser.add_argument("--save_dir", type=str, default='saved_images',
                        help='the data path of the generated adversarial examples')
    parser.add_argument("--ckpt_path", type=str, default='checkpoints/')
    parser.add_argument("--input_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=50)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    idx2label, _ = get_imagenet_dicts()
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    args = get_args()

    test_transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    file_list = glob(os.path.join(args.save_dir, "*.npy"))

    file_dicts = {}
    for path in file_list:
        file_info = os.path.basename(path).split(".")[0]
        if file_info not in file_dicts:
            file_dicts[file_info] = path
    print(file_dicts)

    for key, value in file_dicts.items():
        acc_per_atk_mn = []
        fr_per_atk_mn = []
        for defense_method in ["jpeg", "gaussian_blur", "adv_inception_v3", "env_adv_incep_res_v2", "adv_incep_v2", "tvm", "pixel"]:
            if "jpeg" in defense_method:
                # Transforms
                test_transform = transforms.Compose(
                    [
                        transforms.Lambda(randomJPEGcompression),
                        transforms.ToTensor()
                    ]
                )
                model_name = key.split("_")[0]
            elif "gaussian_blur" in defense_method:
                test_transform = transforms.Compose([
                    transforms.GaussianBlur(kernel_size=(5, 5)),
                    transforms.ToTensor()
                ])
                model_name = key.split("_")[0]
                
            elif "adv_inception_v3" in defense_method:
                model = defense_method
                
            elif "env_adv_incep_res_v2" in defense_method:
                model_name = defense_method

            elif "adv_incep_v2" in defense_method:
                model_name = defense_method
                
            elif "tvm" or "pixel" in defense_method:
                model_name = key.split("_")[0]
            
            dataset = TestDefenseDataset(value, transform=test_transform)
            data_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, num_workers=8, pin_memory=True)
            model = get_model_list(model_name)    
            
            acc, fr, total = test_defense_accuracy(model, data_loader, defense_method,device)
            acc_per_atk_mn.append(acc)
            fr_per_atk_mn.append(fr)

        print_str = f"{key}_{defense_method}: acc: {acc_per_atk_mn}, fr: {fr_per_atk_mn}, total_cnt: {total}"
        print(print_str)
