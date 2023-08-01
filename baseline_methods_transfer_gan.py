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
    fooling_num = torch.sum(adv_pred != clean_pred).item()
    accuracy_adv_pert = torch.sum(adv_pred == pert_pred).item()

    pert_norm = [torch.norm(pi, p=2).item() for pi in pert]

    pcc_value_adv_clean = calculate_pcc(adv_logits, clean_logits)
    pcc_value_adv_pert = calculate_pcc(adv_logits, pert_logits)
    pcc_value_adv_pert_clean = calculate_pcc(adv_logits, clean_pert_logits)

    return adv_pred.cpu().data.numpy(), pert_pred.cpu().data.numpy(), adv_ops_pred.cpu().data.numpy(), \
           accuracy_clean, accuracy_adv, accuracy_op, fooling_num, accuracy_adv_pert,\
           pert_norm, pcc_value_adv_clean, pcc_value_adv_pert, pcc_value_adv_pert_clean


def test_advops_generator(generator, model_list):
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
            "fooling_num":0,
            "accuracy_adv_pert":0,
            "total_samples":0,
        }
    for i, batch in enumerate(validation_loader):

        images, labels = batch
        images, labels = images.to(device), labels.to(device)

        # train the discriminator with fake data
        pert = generator(images)
        adv_images = torch.clamp(images + pert, 0, 1)

        pert = 1.0 / 2 * (pert + 1)
        for key, model in model_list.items():
            with torch.no_grad():
                f_adv = model(adv_images)
                f_clean = model(images)
                f_pert = model(pert)

            adv_pred, pert_pred, adv_ops_pred, \
            acc_clean, acc_adv, acc_op, fooling_num, adv_pert_num, \
            pert_norm, \
            pcc_value_adv_clean, pcc_value_adv_pert, pcc_value_adv_pert_clean = calculate_mertic(pert, labels,
                                                                                                 f_adv, f_clean,
                                                                                                 f_pert)

            if key not in mertics_dict:
                mertics_dict[key] = copy.deepcopy(tmp_dict)
            mertics_dict[key]['adv_cls'].extend(adv_pred)
            mertics_dict[key]['pert_cls'].extend(pert_pred)
            mertics_dict[key]['clean_pert_cls'].extend(adv_ops_pred)

            mertics_dict[key]['acc_clean'] += acc_clean
            mertics_dict[key]['acc_adv'] += acc_adv
            mertics_dict[key]['acc_op'] += acc_op
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
    parser.add_argument("--ckpt_path", type=str, default='checkpoints/ImageNet', help="path of the model checkpoints")
    parser.add_argument("--model_names", nargs='+', type=str,
                        default='resnet50 vgg16 densenet121 resnext wideresnet squeezenet', help="evaluate model list")
    parser.add_argument("--attacks", type=str, default='AdvOps', help="Attack method")
    parser.add_argument("--input_size", type=int, default=224)
    parser.add_argument("--num_filter", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=50)
    parser.add_argument("--num_epochs", type=int, default=100)

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

    model_name_list = args.model_names

    model_list = get_model_list(model_name_list)
    g = OpsAdvGenerator(3, args.num_filter, 3)
    model_name_list = ["resnet50", "vgg16", "densenet121", "resnext", "wideresnet", "mnasnet", "squeezenet"]
    logit_type = ["custom_loss", "mse_loss", "kl_loss"]
    idx1 = 1
    idx2 = 0
    specific_str = f'{model_name_list[idx1]}_{logit_type[idx2]}'

    ckpt_file_dict = {
        "ImageNet_resnet50": f'{args.ckpt_path}/ImageNet_Generator_for_resnet50.pth',
        "ImageNet_vgg16": f'{args.ckpt_path}/ImageNet_Generator_for_vgg16.pth',
        "ImageNet_densenet121": f'{args.ckpt_path}/ImageNet_Generator_for_densenet121.pth',
        "ImageNet_resnext": f'{args.ckpt_path}/ImageNet_Generator_for_resnext50.pth',
        "ImageNet_wideresnet": f'{args.ckpt_path}/ImageNet_Generator_for_wideresnet50.pth',
        "ImageNet_squeezenet": f'{args.ckpt_path}/ImageNet_Generator_for_squeezenet.pth',
    }

    ckpt_file = ckpt_file_dict[f"ImageNet_{args.model_name}"]
    g.load_state_dict(torch.load(ckpt_file))
    g = g.to(device)
    g.eval()
    model_name = model_name_list[0]
    attack_metric_dict = test_advops_generator(g, model_list)
    attack_metric_dict = post_propress_dict_data(attack_metric_dict, model_list)
    print(f"Target model: {model_name}\n Attack: AdvOpsGAN\n", attack_metric_dict)
    result_dict = {
        "attack": "AdvOpsGAN",
        "target_model": model_name,
        "result": attack_metric_dict
    }
    try:
        json_string = json.dumps(result_dict, sort_keys=False, cls=NpEncoder)
        with open(f"baseline_evaluate_result/{model_name}_AdvOpsGAN_{specific_str}.json", 'w') as fw:
            fw.write(json_string)
    except:
        with open(f"baseline_evaluate_result/{model_name}_AdvOpsGAN_{specific_str}.pkl", 'w') as fw:
            pickle.dump(result_dict, fw)
    finally:
        np.save(f"baseline_evaluate_result/{model_name}_AdvOpsGAN_{specific_str}.npy", result_dict)
