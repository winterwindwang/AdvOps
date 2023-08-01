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
from cifar10_models import *
import torchvision


def load_model(archs):
    model_list = {}
    for arch, ckpt_path in archs.items():
        normalize = Normalize(mean=[0.4914, 0.4822, 0.4465],
                              std=[0.2023, 0.1994, 0.2010])
        model = globals()[arch]()
        ckpt = torch.load(ckpt_path)
        model = torch.nn.Sequential(normalize, model)
        model.load_state_dict(ckpt['state_dict'])
        # model = model[1] # only use the model, rather than the normalize
        model.eval()
        model.to(device)
        model_list[arch] = model
    return model_list


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def plot(img1, img1_logit,
         img2, img2_logit,
         img1_img2, img1_img2_logit,
         center_label):
    get_imagenet_dicts()

    fig = plt.figure(figsize=(27, 6))

    ax = plt.subplot(1, 5, 1)
    ax.imshow(img1, vmin=0, vmax=1)
    cl_img_1 = np.argmax(img1_logit)
    img1_label = idx2label[cl_img_1].replace("_", " ")
    ax.set_xlabel("{} ({:.2f}%)".format(img1_label, softmax(img1_logit)[cl_img_1] * 100))
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

    ax = plt.subplot(1, 5, 2)
    ax.imshow(img2, vmin=0, vmax=1)
    cl_img_2 = np.argmax(img2_logit)
    img2_label = idx2label[cl_img_2].replace("_", " ")
    ax.set_xlabel("{}".format(center_label))
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

    ax = plt.subplot(1, 5, 3)
    ax.imshow(img1_img2, vmin=0, vmax=1)
    cl_img_3 = np.argmax(img1_img2_logit)
    img3_label = idx2label[cl_img_3].replace("_", " ")
    ax.set_xlabel("{} ({:.2f}%)".format(img3_label, softmax(img1_img2_logit)[cl_img_3] * 100))
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

    # Lgit plots
    ax = plt.subplot(1, 5, 4)
    ax.scatter(img1_logit, img1_img2_logit, s=2, color="blue")
    ax.set_xlabel("$L_a$")
    ax.set_ylabel("$L_c$", rotation=0)

    # Calculating the covariance matrices
    # logit_mat1 = np.concatenate((img1_logit, img1_img2_logit), axis=0)
    # img1_logit： 干净样本logit输出
    # img1_img2_logit: 对抗样本的logit输出
    logit_mat1 = np.stack((img1_logit, img1_img2_logit), axis=0)
    origin1 = np.mean(logit_mat1, axis=1)

    # Calculate Pearson coefficient
    pcc1 = np.corrcoef(logit_mat1)
    props = dict(facecolor='none', edgecolor='black', pad=5, alpha=0.5)
    ax.text(0.65, 0.05, "PCC: {:.2f}".format(pcc1[0, 1]), transform=ax.transAxes, fontsize=20,
            verticalalignment='bottom', bbox=props)

    ax = plt.subplot(1, 5, 5)
    ax.scatter(img2_logit, img1_img2_logit, s=2, color="red")
    ax.set_xlabel("$L_b$")
    ax.set_ylabel("$L_c$", rotation=0)

    # Calculating the covariance matrices
    # logit_mat2 = np.concatenate((img2_logit, img1_img2_logit), axis=0)
    # img2_logit： 对抗扰动的logit输出
    # img1_img2_logit: 对抗样本的logit输出
    logit_mat2 = np.stack((img2_logit, img1_img2_logit), axis=0)

    # Calculate Pearson coefficient
    pcc2 = np.corrcoef(logit_mat2)
    ax.text(0.65, 0.05, "PCC: {:.2f}".format(pcc2[0, 1]), transform=ax.transAxes, fontsize=20,
            verticalalignment='bottom', bbox=props)

    plt.tight_layout()
    return fig


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
           accuracy_clean, accuracy_adv, accuracy_op, fooling_num, accuracy_adv_pert, \
           pert_norm, pcc_value_adv_clean, pcc_value_adv_pert, pcc_value_adv_pert_clean


def test(model_list, model_name):
    mertics_dict = {}
    tmp_dict = {
        "adv_cls": [],
        "pert_cls": [],
        "clean_pert_cls": [],

        "l2_norms_list": [],
        "pcc_adv_clean": [],
        "pcc_value_adv_pert": [],
        "pcc_value_adv_pert_clean": [],

        "acc_clean": 0,
        "acc_adv": 0,
        "acc_op": 0,
        "fooling_num": 0,
        "accuracy_adv_pert": 0,
        "total_samples": 0,
    }
    # 针对特定网络生成的对抗样本，在其他网络上进行验证
    for i, batch in enumerate(validation_loader):
        images, labels = batch
        images, labels = images.to(device), labels.to(device)

        f_clean = model_list[model_name](images)
        f_clean_pred = torch.argmax(f_clean, dim=1)
        adv_images = atk(images, f_clean_pred.detach())
        pert = adv_images - images
        pert = 1.0 / 2 * (pert + 1)
        # model_name_list = ["resnet50", "vgg16", "densenet121", "resnext", "wideresnet", "mnasnet", "squeezenet"]
        for key, model in model_list.items():
            with torch.no_grad():
                f_adv = model(adv_images)
                f_clean = model(images)
                f_pert = model(pert)

            adv_pred, pert_pred, adv_ops_pred, \
            acc_clean, acc_adv, acc_op, fooling_num, adv_pert_num, \
            pert_norm, \
            pcc_value_adv_clean, pcc_value_adv_pert, pcc_value_adv_pert_clean = calculate_mertic(pert, labels, f_adv,
                                                                                                 f_clean, f_pert)

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


def load_cifar10(args):
    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])
    testset = torchvision.datasets.CIFAR10(root=args.data_dir, train=False,
                                           download=True, transform=transform_test)
    return testset


def get_args():
    parser = argparse.ArgumentParser(description="Args Container")
    parser.add_argument("--data_dir", type=str, default='cifar10_models/data', help="data path of the cifar10")
    parser.add_argument("--attacks", nargs='+', type=str, default='FGSM BIM PGD MIFGSM CW', help="Attack method list")
    parser.add_argument("--ckpt_path", type=str, default='checkpoints/CIFAR10', help="path of the model checkpoints")
    parser.add_argument("--input_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=50)
    parser.add_argument("--num_workers", type=int, default=8)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    idx2label, _ = get_imagenet_dicts()
    args = get_args()

    testset = load_cifar10(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_name_list = {
        'preactresnet18': f"{args.ckpt_path}/CIFAR10_preactresnet18.pth.tar",
        'wideresnet': f"{args.ckpt_path}//CIFAR10_wideresnet.pth.tar",
        "resnet50": f"{args.ckpt_path}/CIFAR10_resnet50.pth.tar",
        "densenet121": f"{args.ckpt_path}/CIFAR10_densenet121.pth.tar",
        "vgg16": f"{args.ckpt_path}/CIFAR10_vgg16.pth.tar",
        "vgg19": f"{args.ckpt_path}/CIFAR10_vgg19.pth.tar"
    }

    model_list = load_model(model_name_list)

    attack_list = args.attacks
    for model_name, model in model_list.items():
        validation_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size,
                                                        shuffle=False, num_workers=args.num_workers)
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
                atk = torchattacks.CW(model, c=1e-4, kappa=0, steps=1000, lr=0.01)  # tensor(15.2147, device='cuda:0')
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
                with open(f"baseline_evaluate_result/cifar10_{model_name}_{attack}.json", 'w') as fw:
                    fw.write(json_string)
            except:
                with open(f"baseline_evaluate_result/cifar10_{model_name}_{attack}.pkl", 'w') as fw:
                    pickle.dump(result_dict, fw)
            finally:
                np.save(f"baseline_evaluate_result/cifar10_{model_name}_{attack}.npy", result_dict)
