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


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def calculate_pcc(x, y):
    """
    Calculate pearsonr mimic `scipy.stats.pearsonr`
    :param x: Logit output (N, num_classes)
    :param y: Logit output (N, num_classes)
    :return: N
    """
    pearson = []
    for i, j in zip(x, y):
        mean_x = torch.mean(i)
        mean_y = torch.mean(j)
        xm = i.sub(mean_x)
        ym = j.sub(mean_y)

        r_num = xm.dot(ym)
        r_den = torch.norm(xm, 2) * torch.norm(ym, 2)
        r_val = r_num / r_den
        pearson.append(r_val.item())
    return pearson


def calculate_mertic(pert, labels, adv_logits, clean_logits, pert_logits):
    """
    Calculate the metric
    :param pert:
    :param labels:
    :param adv_logits:
    :param clean_logits:
    :param pert_logits:
    :return: numpy list
    """
    clean_pert_logits = clean_logits + pert_logits

    clean_pred = torch.argmax(clean_logits, dim=1)
    adv_pred = torch.argmax(adv_logits, dim=1)
    pert_pred = torch.argmax(pert_logits, dim=1)
    adv_ops_pred = torch.argmax(clean_pert_logits, dim=1)

    accuracy_clean = torch.sum(clean_pred == labels).item()
    accuracy_adv = torch.sum(adv_pred == labels).item()
    accuracy_op = torch.sum(adv_pred == adv_ops_pred).item()
    accuracy_op_fr = torch.sum((adv_pred == adv_ops_pred) == (adv_pred != clean_pred)).item()
    fooling_num = torch.sum(adv_pred != clean_pred).item()
    accuracy_adv_pert = torch.sum(adv_pred == pert_pred).item()

    pert_norm = [torch.norm(pi, p=2).item() for pi in pert]

    pcc_value_adv_clean = calculate_pcc(adv_logits, clean_logits)
    pcc_value_adv_pert = calculate_pcc(adv_logits, pert_logits)
    pcc_value_adv_pert_clean = calculate_pcc(adv_logits, clean_pert_logits)

    return adv_pred.cpu().data.numpy(), pert_pred.cpu().data.numpy(), adv_ops_pred.cpu().data.numpy(), \
           accuracy_clean, accuracy_adv, accuracy_op, accuracy_op_fr, fooling_num, accuracy_adv_pert,\
           pert_norm, pcc_value_adv_clean, pcc_value_adv_pert, pcc_value_adv_pert_clean


def test(model_list, model_name):

    mertics_dict = {}
    tmp_dict = {
            "adv_cls":[],
            "pert_cls":[],
            "clean_pert_cls":[],

            "l2_norms_list":[],
            "pcc_adv_clean":[],
            "pcc_value_adv_pert":[],
            "pcc_value_adv_pert_clean":[],

            "acc_clean":0,
            "acc_adv":0,
            "acc_op":0,
            "acc_op_fr":0,
            "fooling_num":0,
            "accuracy_adv_pert":0,
            "total_samples":0,
        }
    # 针对特定网络生成的对抗样本，在其他网络上进行验证
    for i, batch in enumerate(validation_loader):
        images, labels = batch
        images, labels = images.to(device), labels.to(device)
        

        f_clean = model_list[model_name](images)
        f_clean_pred = torch.argmax(f_clean, dim=1)
        adv_images = atk(images, f_clean_pred.detach())
        #################################
        pert = adv_images - images
        pert = 1.0/2 * (pert + 1)
        for key, model in model_list.items():
            with torch.no_grad():
                f_adv = model(adv_images)
                f_clean = model(images)
                f_pert = model(pert)

            adv_pred, pert_pred, adv_ops_pred, \
            acc_clean, acc_adv, acc_op, acc_op_fr, fooling_num, adv_pert_num, \
            pert_norm, \
            pcc_value_adv_clean, pcc_value_adv_pert, pcc_value_adv_pert_clean = calculate_mertic(pert, labels, f_adv, f_clean, f_pert)

            if key not in mertics_dict:
                mertics_dict[key] = copy.deepcopy(tmp_dict)

            mertics_dict[key]['adv_cls'].extend(adv_pred)
            mertics_dict[key]['pert_cls'].extend(pert_pred)
            mertics_dict[key]['clean_pert_cls'].extend(adv_ops_pred)

            mertics_dict[key]['acc_clean'] += acc_clean
            mertics_dict[key]['acc_adv'] += acc_adv
            mertics_dict[key]['acc_op'] += acc_op
            mertics_dict[key]['acc_op_fr'] += acc_op_fr
            mertics_dict[key]['fooling_num'] += fooling_num
            mertics_dict[key]['accuracy_adv_pert'] += adv_pert_num

            mertics_dict[key]['l2_norms_list'].extend(pert_norm)
            mertics_dict[key]['pcc_adv_clean'].extend(pcc_value_adv_clean)
            mertics_dict[key]['pcc_value_adv_pert'].extend(pcc_value_adv_pert)
            mertics_dict[key]['pcc_value_adv_pert_clean'].extend(pcc_value_adv_pert_clean)

            mertics_dict[key]['total_samples'] += images.size(0)

    return mertics_dict


def post_propress_dict_data(data_dict, model_list):
    for key, model in model_list.items():
        # most common top100
        data_dict[key]['adv_cls'] = Counter(data_dict[key]['adv_cls']).most_common(100)
        data_dict[key]['pert_cls'] = Counter(data_dict[key]['pert_cls']).most_common(100)
        data_dict[key]['clean_pert_cls'] = Counter(data_dict[key]['clean_pert_cls']).most_common(100)

        data_dict[key]['acc_clean'] = round(
            data_dict[key]['acc_clean'] * 100 / data_dict[key]['total_samples'], 2)
        data_dict[key]['acc_adv'] = round(
            data_dict[key]['acc_adv'] * 100 / data_dict[key]['total_samples'], 2)
        data_dict[key]['acc_op'] = round(
            data_dict[key]['acc_op'] * 100 / data_dict[key]['total_samples'], 2)
        data_dict[key]['acc_op_fr'] = round(
            data_dict[key]['acc_op_fr'] * 100 / data_dict[key]['total_samples'], 2)
        data_dict[key]['fooling_num'] = round(
            data_dict[key]['fooling_num'] * 100 / data_dict[key]['total_samples'], 2)
        data_dict[key]['accuracy_adv_pert'] = round(
            data_dict[key]['accuracy_adv_pert'] * 100 / data_dict[key]['total_samples'], 2)

        data_dict[key]['l2_norms_list'] = round(np.mean(data_dict[key]['l2_norms_list']), 2)
        data_dict[key]['pcc_adv_clean'] = round(np.mean(data_dict[key]['pcc_adv_clean']), 2)
        data_dict[key]['pcc_value_adv_pert'] = round(np.mean(data_dict[key]['pcc_value_adv_pert']), 2)
        data_dict[key]['pcc_value_adv_pert_clean'] = round(
            np.mean(data_dict[key]['pcc_value_adv_pert_clean']), 2)
    return data_dict


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
        model = torch.nn.Sequential(normalize, model)
        model.eval()
        model.to(device)
        model_list[mn] = model
    return model_list

def get_args():
    parser = argparse.ArgumentParser(description="Args Container")
    parser.add_argument("--data_dir", type=str, default='/mnt/jfs/wangdonghua/dataset/ImageNet/')
    parser.add_argument("--model_names", nargs='+', type=str, default='resnet50 vgg16 densenet121 resnext wideresnet squeezenet', help="evaluate model list")
    parser.add_argument("--attacks", nargs='+', type=str, default='FGSM BIM PGD MIFGSM CW', help="Attack method list")
    parser.add_argument("--input_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=50)
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
    traindir = os.path.join(args.data_dir, 'ImageNet10k')
    valdir = os.path.join(args.data_dir, 'val')
    # dataset
    test_data = ImageFolder(root=valdir, transform=test_transform)

    validation_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_name_list = args.model_name
    model_list = get_model_list(model_name_list)

    attack_list = args.attacks

    for model_name, model in model_list.items():
        for attack in attack_list:
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
            attack_metric_dict = test(model_list, model_name)
            attack_metric_dict = post_propress_dict_data(attack_metric_dict, model_list)

            print(f"Target model: {model_name}\n Attack: {attack}\n", attack_metric_dict)
            result_dict = {
                "attack": attack,
                "target_model": model_name,
                "result": attack_metric_dict
            }
            try:
                json_string = json.dumps(result_dict, sort_keys=False, cls=NpEncoder)
                with open(f"baseline_evaluate_result/{model_name}_{attack}.json", 'w') as fw:
                    fw.write(json_string)
            except:
                with open(f"baseline_evaluate_result/{model_name}_{attack}.pkl", 'w') as fw:
                    pickle.dump(result_dict, fw)
            finally:
                np.save(f"baseline_evaluate_result/{model_name}_{attack}.npy", result_dict)
