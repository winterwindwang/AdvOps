import argparse
import os
import numpy as np
import torch
import torchattacks
from matplotlib import pyplot as plt

from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from utils import *
from generator import OpsAdvGenerator, Discriminator



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
    return np.mean(pearson)




def test():
    pcc_adv_clean = []
    pcc_adv_pert = []
    pcc_pert_clean_adv = []
    cosine_sim_adv_clean = []
    cosine_sim_adv_pert = []
    cosine_sim_pert_clean_adv = []

    for i, batch in enumerate(validation_loader):
        images, labels = batch
        images, labels = images.to(device), labels.to(device)

        adv_images = atk(images, labels)
        pert = adv_images - images

        with torch.no_grad():
            f_adv = model(adv_images)
            f_clean = model(images)
            f_pert = model(pert)

        f_pert_clean = f_clean + f_pert

        pcc_adv_clean.append(calculate_pcc(f_clean, f_adv))
        pcc_adv_pert.append(calculate_pcc(f_pert, f_adv))
        pcc_pert_clean_adv.append(calculate_pcc(f_pert_clean, f_adv))

        cosine_sim_adv_clean.append(torch.mean(torch.cosine_similarity(f_clean, f_adv)).item())
        cosine_sim_adv_pert.append(torch.mean(torch.cosine_similarity(f_pert, f_adv)).item())
        cosine_sim_pert_clean_adv.append(torch.mean(torch.cosine_similarity(f_pert_clean, f_adv)).item())


        # PCC可视化
        # f_clean_logit = f_clean.cpu().detach().numpy()
        # f_adv_logit = f_adv.cpu().detach().numpy()
        # f_pert_logit = f_pert.cpu().detach().numpy()
        #
        # images2show = transforms.ToPILImage()(images[0])
        # adv_images2show = transforms.ToPILImage()(adv_images[0])
        # pert2show = transforms.ToPILImage()(pert[0])
        # fig = plot(images2show, f_clean_logit[0], pert2show,
        #            f_pert_logit[0], adv_images2show, f_adv_logit[0], center_label="Noise")
        # plt.show()

    print("Attack %s:\
          \npcc_adv_clean: %.3f, pcc_adv_pert: %.3f, pcc_pert_clean_adv: %.3f \
        ` \ncosine_sim_adv_clean: %.3f, cosine_sim_adv_pert: %.3f, cosine_sim_pert_clean_adv: %.3f" % (attack,
          np.mean(pcc_adv_clean),
          np.mean(pcc_adv_pert),
          np.mean(pcc_pert_clean_adv),
          np.mean(cosine_sim_adv_clean),
          np.mean(cosine_sim_adv_pert),
          np.mean(cosine_sim_pert_clean_adv)))

def test_advops_generator(generator):
    pcc_adv_clean = []
    pcc_adv_pert = []
    pcc_pert_clean_adv = []
    cosine_sim_adv_clean = []
    cosine_sim_adv_pert = []
    cosine_sim_pert_clean_adv = []

    for i, batch in enumerate(validation_loader):
        images, labels = batch
        images, labels = images.to(device), labels.to(device)

        images, labels = batch
        images, labels = images.to(device), labels.to(device)

        # train the discriminator with fake data
        pert = generator(images)
        adv_images = torch.clamp(images + pert, 0, 1)


        with torch.no_grad():
            f_adv = model(adv_images)
            f_clean = model(images)
            f_pert = model(pert)

        f_pert_clean = f_clean + f_pert

        pcc_adv_clean.append(calculate_pcc(f_clean, f_adv))
        pcc_adv_pert.append(calculate_pcc(f_pert, f_adv))
        pcc_pert_clean_adv.append(calculate_pcc(f_pert_clean, f_adv))

        cosine_sim_adv_clean.append(torch.mean(torch.cosine_similarity(f_clean, f_adv)).item())
        cosine_sim_adv_pert.append(torch.mean(torch.cosine_similarity(f_pert, f_adv)).item())
        cosine_sim_pert_clean_adv.append(torch.mean(torch.cosine_similarity(f_pert_clean, f_adv)).item())




        # PCC可视化
        # f_clean_logit = f_clean.cpu().detach().numpy()
        # f_adv_logit = f_adv.cpu().detach().numpy()
        # f_pert_logit = f_pert.cpu().detach().numpy()
        #
        # images2show = transforms.ToPILImage()(images[0])
        # adv_images2show = transforms.ToPILImage()(adv_images[0])
        # pert2show = transforms.ToPILImage()(pert[0])
        # fig = plot(images2show, f_clean_logit[0], pert2show,
        #            f_pert_logit[0], adv_images2show, f_adv_logit[0], center_label="Noise")
        # plt.show()

    print("Attack %s:\
          \npcc_adv_clean: %.3f, pcc_adv_pert: %.3f, pcc_pert_clean_adv: %.3f \
        ` \ncosine_sim_adv_clean: %.3f, cosine_sim_adv_pert: %.3f, cosine_sim_pert_clean_adv: %.3f" % (attack,
          np.mean(pcc_adv_clean),
          np.mean(pcc_adv_pert),
          np.mean(pcc_pert_clean_adv),
          np.mean(cosine_sim_adv_clean),
          np.mean(cosine_sim_adv_pert),
          np.mean(cosine_sim_pert_clean_adv)))


def get_args():
    parser = argparse.ArgumentParser(description="Args Container")
    parser.add_argument("--data_dir", type=str, default='/mnt/jfs/wangdonghua/dataset/ImageNet/', help="D:/DataSource/ImageNet/")
    parser.add_argument("--ckpt_path", type=str, default='checkpoints/')
    parser.add_argument("--lr_g", type=float, default=0.0002)
    parser.add_argument("--lr_d", type=float, default=0.0002)
    parser.add_argument("--beta1", type=float, default=0.5)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--input_size", type=int, default=224)
    parser.add_argument("--num_filter", type=int, default=64)
    parser.add_argument("--lambda", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=50)
    parser.add_argument("--num_epochs", type=int, default=100)

    args = parser.parse_args()
    return args



if __name__ == "__main__":
    idx2label, _ = get_imagenet_dicts()

    args = get_args()

    train_transform = transforms.Compose([
        transforms.Resize(256),
        # transforms.Resize(299), # inception_v3
        transforms.RandomCrop(args.input_size),
        # transforms.RandomRotation([-5, 5]),
        # transforms.RandomHorizontalFlip(),
        # transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
    ])

    test_transform = transforms.Compose([
        transforms.Resize(256),
        # transforms.Resize(299), # inception_v3
        transforms.CenterCrop(args.input_size),
        transforms.ToTensor(),
    ])
    traindir = os.path.join(args.data_dir, 'ImageNet10k')
    valdir = os.path.join(args.data_dir, 'val')
    # dataset
    train_data = ImageFolder(root=traindir, transform=train_transform)
    test_data = ImageFolder(root=valdir, transform=test_transform)

    validation_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = models.resnet50(pretrained=True)
    normalize = Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225])
    model = torch.nn.Sequential(normalize, model)
    model.eval()
    model.to(device)

    isbaseline = True

    if isbaseline:
        for attack in ["FGSM", "PGD", "BIM", "MIFGSM", "CW"]:
            if attack == "PGD":
                atk = torchattacks.PGD(model, eps=10 / 255)
            elif attack == "FGSM":
                atk = torchattacks.FGSM(model, eps=10 / 255)
            elif attack == "BIM":
                atk = torchattacks.BIM(model, eps=10 / 255)
            elif attack == "MIFGSM":
                atk = torchattacks.MIFGSM(model, eps=10 / 255)
            elif attack == "CW":
                atk = torchattacks.CW(model)
            test()
    else:
        g = OpsAdvGenerator(3, args.num_filter, 3)
        d = Discriminator(3, args.num_filter, 1)
        g = g.to(device)
        d = d.to(device)
        # g.normal_weight_init(mean=0.0, std=0.02)
        # d.normal_weight_init(mean=0.0, std=0.02)
        g.load_state_dict(torch.load("checkpoints/Lastest_Generator_epoch_v1.pth"))
        g.eval()
        test_advops_generator(g)
