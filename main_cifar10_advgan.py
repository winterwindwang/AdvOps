from email.mime import image
from statistics import mode
import numpy as np
from sympy import im

from generator_cifar10 import Generator, Discriminator
from torchvision import transforms
from cifar10_models import *
from torchvision.datasets import ImageFolder
import os
from torch.autograd import Variable
import torch.nn.functional as F
import torch
from utils import *
import argparse
import time
import torchvision
import torch.nn as nn
from tensorboardX import SummaryWriter


def train_g(epoch, logit_type, loss_type='cw'):
    LongTensor = torch.LongTensor
    acc_cnt = 0
    adv_ops_cnt = 0
    total_cnt = 0
    acc_cnt = 0
    fool_cnt = 0
    acc_adv_cnt = 0
    adv_pert_cnt = 0
    g.train()
    for i, batch in enumerate(train_loader):
        images, labels = batch
        images, labels = images.to(device), labels.to(device)
        random_labels = Variable(LongTensor(np.random.randint(0, 10, images.size(0)))).to(device)

        # train the discriminator with fake data
        perturbation = g(images)
        # perturbation = torch.clamp(g(images), -eps, eps)

        # first constrain the range of the perturbation into [-eps, eps]
        # perturbation = torch.clamp(perturbation, -eps, eps)

        adv_image = images + perturbation
        # then constrain the adversarial examples into [0, 1]
        adv_image = torch.clamp(images + perturbation, 0, 1)
        f_adv = model(adv_image)
        # Generator
        
        f_perturbation = model((1.0 / 2) * (perturbation + 1))
        
        f_clean = model(images)
        f_clean_pred = torch.argmax(f_clean, dim=1).detach()

        # advops loss (f_clean_pert[clean_cls] - f_clean_pert[adv_cls])
        if logit_type == "mse_loss":
            # mse loss
            pass
            # loss_logit = mse_loss(f_adv, f_perturbation + f_clean)
        elif logit_type == "kl_loss":
            # kl div loss
            pass
            # loss_logit = kl_loss(torch.log_softmax(f_adv, dim=1), torch.softmax(f_perturbation + f_clean, dim=1))
        elif logit_type == "custom_loss":
            # custous loss
            pass
            # f_pert_clean = f_perturbation + f_clean
            # # f_clean_pred = torch.argmax(f_clean, dim=1).detach()
            # f_adv_pred = torch.argmax(f_adv, dim=1).detach()
            # one_hot_labels = torch.eye(len(f_clean[0]))[f_clean_pred].to(device)
            # one_hot_adv_labels = torch.eye(len(f_clean[0]))[f_adv_pred].to(device)

            # pert_logit = torch.masked_select(f_pert_clean,
            #                                  one_hot_adv_labels.bool())  # get the second largest logit in clean+pred
            # clean_logit = torch.masked_select(f_pert_clean, one_hot_labels.bool())
            # loss_logit = torch.clamp((clean_logit - pert_logit), min=0).sum()
        # logit_diff = clean_logit - pert_logit
        # loss_logit = torch.max(logit_diff, -0.2 * torch.ones_like(logit_diff)).sum() # attack strength

        # loss_logit.backward(retain_graph=True)

        # calculate perturbation norm
        C = 0.1
        loss_perturb = torch.mean(torch.norm(perturbation.view(perturbation.shape[0], -1), 2, dim=1))
        # loss_perturb = torch.max(loss_perturb - C, torch.zeros(1, device=self.device))

        if loss_type == "cw":
            # cal adv loss
            probs_model = F.softmax(f_adv, dim=1)
            onehot_labels = torch.eye(10, device=device)[f_clean_pred]

            # C&W loss function
            real = torch.sum(onehot_labels * probs_model, dim=1)
            other, _ = torch.max((1 - onehot_labels) * probs_model - onehot_labels * 10000, dim=1)
            zeros = torch.zeros_like(other)
            loss_adv = torch.max(real - other, zeros)
            loss_adv = torch.sum(loss_adv)
        else:
            loss_adv = -ce_loss(f_adv, f_clean_pred)

        adv_lambda = 10
        pert_lambda = 1
        # logit_lambda = 10

        loss_adv = adv_lambda * loss_adv
        loss_perturb = pert_lambda * loss_perturb
        if epoch > 20:
            loss_G = loss_adv + 20 * loss_perturb  # advgan
        else:
            loss_G = loss_adv + loss_perturb
        g_optimizer.zero_grad()
        loss_G.backward()
        g_optimizer.step()

        # printing parameters
        f_adv_pred = torch.argmax(f_adv, dim=1)
        f_pert_pred = torch.argmax(f_perturbation, dim=1)
        f_clean_pred = torch.argmax(f_clean, dim=1)
        f_clean_pert_pred = torch.argmax(f_clean + f_perturbation, dim=1)

        adv_ops_cnt += (f_clean_pert_pred == f_adv_pred).sum().item()
        acc_cnt += (f_clean_pred == labels).sum().item()
        acc_adv_cnt += (f_adv_pred == labels).sum().item()
        fool_cnt += (f_adv_pred != f_clean_pred).sum().item()
        adv_pert_cnt += (f_adv_pred == f_pert_pred).sum().item()
        total_cnt += images.size(0)

        transforms.ToPILImage()(adv_image[0]).save(f"{save_dir}/image_adv_{epoch}.png")
        transforms.ToPILImage()((1.0 / 2) * (perturbation[0] + 1)).save(f"{save_dir}/image_pert_{epoch}.png")
        transforms.ToPILImage()(images[0]).save(f"{save_dir}/image_clean_{epoch}.png")


        acc_clean = (acc_cnt * 100) / total_cnt
        acc_adv = (acc_adv_cnt * 100) / total_cnt
        acc_adv_ops = (adv_ops_cnt * 100) / total_cnt
        acc_adv_pert = (adv_pert_cnt * 100) / total_cnt
        fr = (fool_cnt * 100) / total_cnt

        # wirter.add_scalars("training/loss", {
        #     "loss_adv": loss_adv, 
        #     "loss_advops": loss_logit, 
        #     "loss_pert": loss_perturb, 
        # }, epoch + i)
        wirter.add_scalar("training/loss_generator", loss_G, epoch + i)
        wirter.add_scalar("training/loss_adv", loss_adv, epoch + i)
        # wirter.add_scalar("training/loss_advops", loss_logit, epoch + i)s
        wirter.add_scalar("training/loss_pert", loss_perturb, epoch + i)
        
        wirter.add_scalar("training/acc_clean", acc_clean, epoch + i)
        wirter.add_scalar("training/acc_adv", acc_adv, epoch + i)
        wirter.add_scalar("training/acc_advops", acc_adv_ops, epoch + i)
        wirter.add_scalar("training/acc_adv_pert", acc_adv_pert, epoch + i)
        wirter.add_scalar("training/fr", fr, epoch + i)
        # wirter.add_scalars("training/metrics", {
        #     "acc_clean": acc_clean, 
        #     "acc_adv": acc_adv, 
        #     "acc_advops": acc_adv_ops, 
        #     "acc_adv_pert": acc_adv_pert, 
        #     "fr": fr
            
        # }, epoch + i)
        print(
            "epoch %d: loss_perturb: %.3f, loss_adv: %.3f, adv_ops: %.3f, acc_clean: %.3f, acc_adv: %.3f, fr_adv: %.3f, acc_adv_pert: %.3f" % (
            epoch, loss_perturb.item(), loss_adv.item(), acc_adv_ops, acc_clean, acc_adv, fr, acc_adv_pert))




# def train_gan(epoch):
#     for i, batch in enumerate(train_loader):
#         images, labels = batch
#         images, labels = images.to(device), labels.to(device)

#         # train the discriminator with real data
#         D_real = d(images).squeeze()
#         real_target = Variable(torch.ones(D_real.size()).to(device))
#         D_real_loss = bce_loss(D_real, real_target)

#         # train the discriminator with fake data
#         perturbation = g(images)
#         adv_image = images + perturbation
#         D_fake = d(adv_image.detach()).squeeze()
#         fake_target = Variable(torch.ones(D_fake.size()).to(device))
#         D_fake_loss = bce_loss(D_fake, fake_target)

#         # backward
#         d_optimizer.zero_grad()
#         D_loss = (D_fake_loss + D_real_loss) * 0.5
#         D_loss.backward()
#         d_optimizer.step()

#         # Generator
#         D_fake_descision = d(adv_image).squeeze()
#         G_fake_loss = bce_loss(D_fake_descision, real_target)

#         g_optimizer.zero_grad()
#         G_fake_loss.backward(retain_graph=True)

#         f_adv = model(adv_image)
#         f_perturbation = model(perturbation)
#         f_clean = model(images)
#         loss_logit = mse_loss(f_adv, (f_perturbation + f_clean))

#         # loss_logit.backward(retain_graph=True)

#         # calculate perturbation norm
#         C = 0.1
#         loss_perturb = torch.mean(torch.norm(perturbation.view(perturbation.shape[0], -1), 2, dim=1))
#         # loss_perturb = torch.max(loss_perturb - C, torch.zeros(1, device=self.device))

#         # cal adv loss
#         probs_model = F.softmax(f_adv, dim=1)
#         onehot_labels = torch.eye(1000, device=device)[labels]

#         # C&W loss function
#         real = torch.sum(onehot_labels * probs_model, dim=1)
#         other, _ = torch.max((1 - onehot_labels) * probs_model - onehot_labels * 10000, dim=1)
#         zeros = torch.zeros_like(other)
#         loss_adv = torch.max(real - other, zeros)
#         loss_adv = torch.sum(loss_adv)

#         adv_lambda = 10
#         pert_lambda = 1
#         loss_G = loss_logit #+ adv_lambda * loss_adv + pert_lambda * loss_perturb

#         loss_G.backward()
#         g_optimizer.step()

#         print("epoch %d:\nloss_D: %.3f, loss_G_fake: %.3f, loss_feat: %.3f,\
#                     \nloss_perturb: %.3f, loss_adv: %.3f, \n" %
#               (epoch, D_loss.item(), G_fake_loss.item(), loss_logit.item(),
#                loss_perturb.item(), loss_adv.item()))

@torch.no_grad()
def test(epoch):
    acc_cnt = 0
    adv_ops_cnt = 0
    total_cnt = 0
    acc_cnt = 0
    fool_cnt = 0
    acc_adv_cnt = 0
    adv_pert_cnt = 0
    g.eval()
    for i, batch in enumerate(validation_loader):
        images, labels = batch
        images, labels = images.to(device), labels.to(device)

        # train the discriminator with fake data
        perturbation = g(images)
        # perturbation = torch.clamp(g(images), -eps, eps)
        adv_image = torch.clamp(images + perturbation, 0, 1)

        # Generator
        f_adv = model(adv_image)
        f_perturbation = model((1.0 / 2) * (perturbation + 1))
        f_clean = model(images)

        f_adv_pred = torch.argmax(f_adv, dim=1)
        f_pert_pred = torch.argmax(f_perturbation, dim=1)
        f_clean_pred = torch.argmax(f_clean, dim=1)
        f_clean_pert_pred = torch.argmax(f_clean + f_perturbation, dim=1)

        adv_ops_cnt += (f_clean_pert_pred == f_adv_pred).sum().item()
        acc_cnt += (f_clean_pred == labels).sum().item()
        acc_adv_cnt += (f_adv_pred == labels).sum().item()
        fool_cnt += (f_adv_pred != f_clean_pred).sum().item()
        adv_pert_cnt += (f_adv_pred == f_pert_pred).sum().item()
        total_cnt += images.size(0)


    acc_clean = (acc_cnt * 100) / total_cnt
    acc_adv = (acc_adv_cnt * 100) / total_cnt
    acc_adv_ops = (adv_ops_cnt * 100) / total_cnt
    acc_adv_pert = (adv_pert_cnt * 100) / total_cnt
    fr = (fool_cnt * 100) / total_cnt

    # wirter.add_scalars("valid/metrics", {
    #     "acc_clean": acc_clean, 
    #     "acc_adv": acc_adv, 
    #     "acc_advops": acc_adv_ops, 
    #     "acc_adv_pert": acc_adv_pert, 
    #     "fr": fr
    # }, epoch)        
    wirter.add_scalar("training/acc_clean", acc_clean, epoch + i)
    wirter.add_scalar("training/acc_adv", acc_adv, epoch + i)
    wirter.add_scalar("training/acc_advops", acc_adv_ops, epoch + i)
    wirter.add_scalar("training/acc_adv_pert", acc_adv_pert, epoch + i)
    wirter.add_scalar("training/fr", fr, epoch + i)


    print("epoch %d:\n, clean_acc: %.3f,\
                \nadv_acc: %.3f, fooling_rate: %.3f, acc_ops: %.3f\n" % (epoch,
                                                                         acc_clean,
                                                                         acc_adv,
                                                                         fr,
                                                                         acc_adv_ops))

    return acc_clean, acc_adv, fr, acc_adv_ops


def get_args():
    parser = argparse.ArgumentParser(description="Args Container")
    parser.add_argument("--data_dir", type=str, default='/mnt/jfs/wangdonghua/dataset/ImageNet/')
    parser.add_argument("--ckpt_path", type=str, default='checkpoints/')
    parser.add_argument("--model_name", type=str, default='vgg16')
    parser.add_argument("--lr_g", type=float, default=0.0002)
    parser.add_argument("--lr_d", type=float, default=0.0002)
    parser.add_argument("--beta1", type=float, default=0.5)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--input_size", type=int, default=224)
    parser.add_argument("--num_filter", type=int, default=64)
    parser.add_argument("--lambda", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--num_epochs", type=int, default=60)
    parser.add_argument("--num_workers", type=int, default=8)

    args = parser.parse_args()
    return args


def load_model(arch, ckpt_path):
    normalize = Normalize(mean=[0.4914, 0.4822, 0.4465],
                        std=[0.2023, 0.1994, 0.2010])
    model = globals()[arch]()
    ckpt = torch.load(ckpt_path)
    model = torch.nn.Sequential(normalize, model)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    model.to(device)
    return model


def load_cifar10(args):
    transform_train = transforms.Compose([
        # transforms.RandomCrop(32, padding=4),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    trainset = torchvision.datasets.CIFAR10(root='/mnt/jfs/wangdonghua/pythonpro/AdvOps/cifar10_models/data', train=True,
                                            download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                              shuffle=True, num_workers=args.num_workers)

    transform_test = transforms.Compose([
        # transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
    ])
    testset = torchvision.datasets.CIFAR10(root='cifar10_models/data', train=False,
                                           download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size,
                                             shuffle=False, num_workers=args.num_workers)
    return trainloader, testloader

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

if __name__ == "__main__":
    args = get_args()

    train_loader,validation_loader = load_cifar10(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_name_list = {
        'preactresnet18':"/mnt/jfs/wangdonghua/pythonpro/AdvOps/cifar10_models/checkpoints/preactresnet18.pth.tar",
        'wideresnet':"/mnt/jfs/wangdonghua/pythonpro/AdvOps/cifar10_models/checkpoints/wideresnet.pth.tar",
        "resnet50": "/mnt/jfs/wangdonghua/pythonpro/AdvOps/cifar10_models/checkpoints/resnet50.pth.tar",
        "resnext29_16x64d": "/mnt/jfs/wangdonghua/pythonpro/AdvOps/cifar10_models/checkpoints/resnext29_16x64d.pth.tar",
        "se_resnext29_16x64d": "/mnt/jfs/wangdonghua/pythonpro/AdvOps/cifar10_models/checkpoints/se_resnext29_16x64d.pth.tar",
        "densenet121": "/mnt/jfs/wangdonghua/pythonpro/AdvOps/cifar10_models/checkpoints/densenet121.pth.tar",
        "vgg16": "/mnt/jfs/wangdonghua/pythonpro/AdvOps/cifar10_models/checkpoints/vgg16.pth.tar",
        "vgg19": "/mnt/jfs/wangdonghua/pythonpro/AdvOps/cifar10_models/checkpoints/vgg19.pth.tar"
    }
    # for model_name, ckpt in model_name_list.items():
    #     model = load_model(model_name, ckpt_path)
    
    model = load_model(args.model_name, model_name_list[args.model_name])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    g = Generator(3, 3)
    g = g.to(device)
    g.apply(weights_init)


    # g_optimizer = torch.optim.Adam(g.parameters(), lr=args.lr_g, betas=(args.beta1, args.beta2))
    g_optimizer = torch.optim.RMSprop(g.parameters(), lr=args.lr_g, weight_decay=0.99, momentum=0.01)

    CosineLR = torch.optim.lr_scheduler.CosineAnnealingLR(g_optimizer, T_max=args.num_epochs, eta_min=0)

    # Loss functions
    bce_loss = torch.nn.BCELoss()
    ce_loss = torch.nn.CrossEntropyLoss()
    mse_loss = torch.nn.MSELoss()
    l1_loss = torch.nn.L1Loss()
    kl_loss = torch.nn.KLDivLoss()

    eps_nomial = 10.0
    eps = eps_nomial / 255

    if args.model_name in ["resnext29_16x64d", "se_resnext29_16x64d"]:
        args.batch_size = 4

    time_str = time.strftime("%m-%d-%H-%M", time.localtime())
    save_dir = f"saved_images/EXP_NAME_{time_str}_eps{int(eps_nomial)}"

    ckpt_save_dir = f"{args.ckpt_path}/EXP_NAME_{time_str}_eps{int(eps_nomial)}"
    if not os.path.exists(ckpt_save_dir):
        os.makedirs(ckpt_save_dir)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # logit_type = ["custom_loss", "mse_loss", "kl_loss"]
    logit_type = ["custom_loss"]

    for loss_type in logit_type:
        time_str = time.strftime("%m-%d-%H-%M", time.localtime())
        wirter = SummaryWriter(f"runs/EXP_{time_str}_{args.model_name}_{loss_type}")

        specific_info = loss_type
        print(f"Model: {args.model_name}, {loss_type}")
        # record metrics
        best_acc_adv = np.inf
        best_fooling_rate = 0
        best_acc_advops = 0
        for epoch in range(1, args.num_epochs + 1):
            train_g(epoch, loss_type)
            if epoch % 5 == 0:
                acc, acc_adv, fooling_rate, acc_advops = test(epoch)

                if fooling_rate > 70 and best_acc_advops < acc_advops:
                    best_acc_advops = acc_advops
                    netG_file_name = f'{ckpt_save_dir}/CIFAR10_Best_advops_acc_Generator_for_{args.model_name}_{specific_info}.pth'
                    torch.save(g.state_dict(), netG_file_name)
                if best_fooling_rate < fooling_rate:
                    best_fooling_rate = fooling_rate
                    netG_file_name = f'{ckpt_save_dir}/CIFAR10_Best_fooling_rate_Generator_for_{args.model_name}_{specific_info}.pth'
                    torch.save(g.state_dict(), netG_file_name)

                if best_acc_adv < 30 and best_acc_adv > acc_adv:
                    best_acc_adv = acc_adv
                    netG_file_name = f'{ckpt_save_dir}/CIFAR10_Best_adv_acc_Generator_for_{args.model_name}_{specific_info}_epoch{str(epoch)}.pth'
                    torch.save(g.state_dict(), netG_file_name)
            CosineLR.step()
            if epoch % (20 + 1) == 0:
                netG_file_name = f'{ckpt_save_dir}/CIFAR10_Generator_for_{args.model_name}_{specific_info}_epoch{str(epoch)}.pth'
                torch.save(g.state_dict(), netG_file_name)

        netG_file_name = f'{ckpt_save_dir}/CIFAR10_Lastest_Generator_for_{args.model_name}_{specific_info}_epoch{str(epoch)}.pth'
        torch.save(g.state_dict(), netG_file_name)
        wirter.close()
    print("Finished!!")