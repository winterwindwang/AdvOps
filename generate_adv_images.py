import argparse
import os
import pickle

import numpy as np
import torch
import torchattacks
from matplotlib import pyplot as plt

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



def test(save_dir):
    cnt = 0
    data_info = []
    for i, batch in enumerate(validation_loader):
        images, labels = batch
        images, labels = images.to(device), labels.to(device)

        adv_images = atk(images, labels)
        if cnt > 1000:
            break
        cnt += images.size(0)
        for j, img in enumerate(adv_images):
            filename = f"{save_dir}/adv_images_{i}_{j}.png"
            transforms.ToPILImage()(img).save(filename)
            data_info.append((filename, labels[j].item()))
    return data_info

def test_advops_generator(geneator, save_dir):
    cnt = 0
    data_info = []
    for i, batch in enumerate(validation_loader):

        images, labels = batch
        images, labels = images.to(device), labels.to(device)

        # train the discriminator with fake data
        pert = geneator(images)
        adv_images = torch.clamp(images + pert, 0, 1)

        if cnt > 1000:
            break
        cnt += images.size(0)
        for j, img in enumerate(adv_images):
            filename = f"{save_dir}/adv_images_{i}_{j}.png"
            transforms.ToPILImage()(img).save(filename)
            data_info.append((filename, labels[j].item()))
    return data_info


def get_model_list(model_name):
    normalize = Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225])
    model_list = {}
    for mn in model_name:
        if "resnet50" in mn.lower():
            model = models.resnet50(pretrained=True)
        elif "vgg16" in mn.lower():
            model = models.vgg16(pretrained=True)
        elif "densenet121" in mn.lower():
            model = models.densenet121(pretrained=True)
        elif "resnext" in mn.lower():
            model = models.resnext50_32x4d(pretrained=True)
        elif "wideresnet" in mn.lower():
            model = models.wide_resnet50_2(pretrained=True)
        elif "mnasnet" in mn.lower():
            model = models.mnasnet1_0(pretrained=True)
        elif "squeezenet" in mn.lower():
            model = models.squeezenet1_0(pretrained=True)
        elif "mlp" in model_name:
            model = timm.create_model('mixer_b16_224', pretrained=True)
        elif "adv_inception_v3" in args.defense_strategy:
            model = timm.create_model('adv_inception_v3', pretrained=True)
        elif "env_adv_incep_res_v2" in args.defense_strategy:
            model = timm.create_model('ens_adv_inception_resnet_v2', pretrained=True) #, num_classes=0, global_pool='')
        elif "adv_incep_v2" in args.defense_strategy:
            model = timm.create_model("inception_resnet_v2", pretrained=True)
        elif "tvm" or "pixel" in args.defense_strategy:
            model = torchvision.models.vgg16(pretrained=True)
        model = torch.nn.Sequential(normalize, model)
        model.eval()
        model.to(device)
        model_list[mn] = model
    return model_list


def get_args():
    parser = argparse.ArgumentParser(description="Args Container")
    parser.add_argument("--data_dir", type=str, default='dataset/ImageNet/', help="data path for ImageNet")
    parser.add_argument("--save_dir", type=str, default='saved_images')
    parser.add_argument("--ckpt_path", type=str, default='checkpoints/ImageNet', help="path of the model checkpoints")
    parser.add_argument("--model_names", nargs='+', type=str, default='resnet50 vgg16 densenet121 resnext wideresnet squeezenet', help="evaluate model list")
    parser.add_argument("--attacks", nargs='+', type=str, default='FGSM BIM PGD MIFGSM CW', help="Attack method list")
    parser.add_argument("--input_size", type=int, default=224)
    parser.add_argument("--num_filter", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=50)
    parser.add_argument("--isbaseline", action="store_true", default=True)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    idx2label, _ = get_imagenet_dicts()
    args = get_args()

    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(args.input_size),
        transforms.ToTensor(),
    ])
    valdir = os.path.join(args.data_dir, 'val')
    # dataset
    test_data = ImageFolder(root=valdir, transform=test_transform)

    validation_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_name_list = ["resnet50", "vgg16", "densenet121", "resnext", "wideresnet", "squeezenet"]
    model_list = get_model_list(model_name_list)

    isbaseline = False
    if isbaseline:
        for model_name, model in model_list.items():
            for attack in ["FGSM", "PGD", "BIM", "MIFGSM", "CW"]:
                if attack == "PGD":
                    atk = torchattacks.PGD(model, eps=10.0 / 255)
                elif attack == "FGSM":
                    atk = torchattacks.FGSM(model, eps=10.0 / 255)
                elif attack == "BIM":
                    atk = torchattacks.BIM(model, eps=10.0 / 255)
                elif attack == "MIFGSM":
                    atk = torchattacks.MIFGSM(model, eps=10.0 / 255)
                elif attack == "CW":
                    atk = torchattacks.CW(model)
                save_dir = f"{args.save_dir}/{model_name}_{attack}_adv_images"
                if not os.path.exists(save_dir):
                    os.mkdir(save_dir)
                data_info = test(save_dir)
                data_info_saved_name = f"{args.save_dir}/{model_name}_{attack}_adv_images"
                np.save(data_info_saved_name, data_info)
                print(f"{model_name}_{attack} is done. File saved in {data_info_saved_name}")
    else:
        g = OpsAdvGenerator(3, args.num_filter, 3)
        model_name_dict = {
            "ImageNet_resnet50": f'{args.ckpt_path}/ImageNet_Generator_for_resnet50.pth',
            "ImageNet_vgg16": f'{args.ckpt_path}/ImageNet_Generator_for_vgg16.pth',
            "ImageNet_densenet121": f'{args.ckpt_path}/ImageNet_Generator_for_densenet121.pth',
            "ImageNet_resnext": f'{args.ckpt_path}/ImageNet_Generator_for_resnext50.pth',
            "ImageNet_wideresnet": f'{args.ckpt_path}/ImageNet_Generator_for_wideresnet50.pth',
            "ImageNet_squeezenet": f'{args.ckpt_path}/ImageNet_Generator_for_squeezenet.pth',
        }
        for model_name, ckpt_path in model_name_dict.items():
            g.load_state_dict(torch.load(ckpt_path))
            g = g.to(device)
            g.eval()

            save_dir = f"{args.save_dir}/{model_name}_AdvOps_adv_images"
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            data_info = test_advops_generator(g, save_dir)
            data_info_saved_name = f"{args.save_dir}/{model_name}_AdvOps_adv_images"
            np.save(data_info_saved_name, data_info)
            print(f"{model_name}_AdvOps is done. File saved in {data_info_saved_name}")
           
    print("Finished!!")