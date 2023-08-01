from email.mime import image
from statistics import mode
import numpy as np

from generator import OpsAdvGenerator, Discriminator, OpsAdvGenerator_PixelShuffle
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
import os
from torch.autograd import Variable
import torch.nn.functional as F
import torch
from utils import *
import argparse
import time
from tensorboardX import SummaryWriter


def train_g(epoch, generator, loss_type='cw'):
    LongTensor = torch.LongTensor
    acc_cnt = 0
    adv_ops_cnt = 0
    total_cnt = 0
    acc_cnt = 0
    fool_cnt = 0
    acc_adv_cnt = 0
    adv_pert_cnt = 0
    advop_fr_cnt = 0
    g.train()
    for i, batch in enumerate(train_loader):
        images, labels = batch
        images, labels = images.to(device), labels.to(device)
        random_labels =  Variable(LongTensor(np.random.randint(0, 1000, images.size(0)))).to(device)

        # train the discriminator with fake data
        perturbation = generator(images)
        # perturbation = torch.clamp(g(images), -eps, eps)
        
        # first constrain the range of the perturbation into [-eps, eps] 
        # perturbation = torch.clamp(perturbation, -eps, eps)
        
        adv_image = images + perturbation
        # then constrain the adversarial examples into [0, 1]
        adv_image = torch.clamp(images + perturbation, 0, 1)

        # Generator
        f_adv = model(adv_image)
        f_perturbation = model((1.0/2)*(perturbation + 1))
        f_clean = model(images)



        # calculate perturbation norm
        C = 0.1
        loss_perturb = torch.mean(torch.norm(perturbation.view(perturbation.shape[0], -1), 2, dim=1))
        # loss_perturb = torch.max(loss_perturb - C, torch.zeros(1, device=self.device))

        if loss_type=="cw":
            # cal adv loss
            probs_model = F.softmax(f_adv, dim=1)
            onehot_labels = torch.eye(1000, device=device)[labels]

            # C&W loss function
            real = torch.sum(onehot_labels * probs_model, dim=1)
            other, _ = torch.max((1 - onehot_labels) * probs_model - onehot_labels * 10000, dim=1)
            zeros = torch.zeros_like(other)
            loss_adv = torch.max(real - other, zeros)
            loss_adv = torch.sum(loss_adv)
        else:
            loss_adv = -ce_loss(f_adv, labels)
 
        adv_lambda = 10
        pert_lambda = 1
        logit_lambda = 10
        
        # loss_logit = logit_lambda * loss_logit
        loss_adv = adv_lambda * loss_adv
        loss_perturb = pert_lambda * loss_perturb
        if epoch > 20:
            loss_G = loss_adv + 10 * loss_perturb  # advgan
        else:
            loss_G = loss_adv + loss_perturb
        # loss_G = adv_lambda * loss_adv + pert_lambda * loss_perturb
        g_optimizer.zero_grad()
        loss_G.backward()
        g_optimizer.step()

        # printing parameters
        f_adv_pred = torch.argmax(f_adv, dim=1)
        f_pert_pred = torch.argmax(f_perturbation, dim=1)
        f_clean_pred = torch.argmax(f_clean, dim=1)
        f_clean_pert_pred = torch.argmax(f_clean+f_perturbation, dim=1)

        adv_ops_cnt += (f_clean_pert_pred==f_adv_pred).sum().item()
        acc_cnt += (f_clean_pred == labels).sum().item()
        acc_adv_cnt += (f_adv_pred == labels).sum().item()
        fool_cnt += (f_adv_pred != labels).sum().item()
        adv_pert_cnt += (f_adv_pred == f_pert_pred).sum().item()
        advop_fr_cnt += torch.sum((f_adv_pred == f_clean_pert_pred) == (f_adv_pred != f_clean_pred)).item()


        total_cnt += images.size(0)
        
        transforms.ToPILImage()(adv_image[0]).save(f"{save_dir}/image_adv_{epoch}.png")
        transforms.ToPILImage()((1.0/2)*(perturbation[0] + 1)).save(f"{save_dir}/image_pert_{epoch}.png")
        transforms.ToPILImage()(images[0]).save(f"{save_dir}/image_clean_{epoch}.png")

        acc_clean = (acc_cnt * 100) / total_cnt
        acc_adv = (acc_adv_cnt * 100) / total_cnt
        acc_adv_ops = (adv_ops_cnt * 100) / total_cnt
        acc_adv_pert = (adv_pert_cnt * 100) / total_cnt
        fr = (fool_cnt * 100) / total_cnt
        adavops_fr = (advop_fr_cnt * 100) / total_cnt

        wirter.add_scalar("training/loss_generator", loss_G, epoch + i)
        wirter.add_scalar("training/loss_adv", loss_adv, epoch + i)
        # wirter.add_scalar("training/loss_advops", loss_logit, epoch + i)
        wirter.add_scalar("training/loss_pert", loss_perturb, epoch + i)
        
        wirter.add_scalar("training/acc_clean", acc_clean, epoch + i)
        wirter.add_scalar("training/acc_adv", acc_adv, epoch + i)
        wirter.add_scalar("training/acc_advops", acc_adv_ops, epoch + i)
        wirter.add_scalar("training/acc_adv_pert", acc_adv_pert, epoch + i)
        wirter.add_scalar("training/acc_advops_fr", adavops_fr, epoch + i)
        wirter.add_scalar("training/fr", fr, epoch + i)

        print(
            "epoch %d: loss_perturb: %.3f, loss_adv: %.3f, adv_ops: %.3f, acc_clean: %.3f, acc_adv: %.3f, fr_adv: %.3f, acc_adv_pert: %.3f" % (
            epoch, loss_perturb.item(), loss_adv.item(), acc_adv_ops, acc_clean, acc_adv, fr, acc_adv_pert))
    


@torch.no_grad()
def test(epoch, generator):
    acc_cnt = 0
    adv_ops_cnt = 0
    total_cnt = 0
    acc_cnt = 0
    fool_cnt = 0
    acc_adv_cnt = 0
    adv_pert_cnt = 0
    advop_fr_cnt = 0
    g.eval()
    for i, batch in enumerate(validation_loader):
        images, labels = batch
        images, labels = images.to(device), labels.to(device)

        # train the discriminator with fake data
        perturbation = generator(images)
        # perturbation = torch.clamp(g(images), -eps, eps)
        adv_image = torch.clamp(images + perturbation, 0, 1)

        # Generator
        f_adv = model(adv_image)
        f_perturbation = model((1.0/2)*(perturbation + 1))
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
        advop_fr_cnt += torch.sum((f_adv_pred == f_clean_pert_pred) == (f_adv_pred != f_clean_pred)).item()

    acc_clean = (acc_cnt * 100) / total_cnt
    acc_adv = (acc_adv_cnt * 100) / total_cnt
    acc_adv_ops = (adv_ops_cnt * 100) / total_cnt
    acc_adv_pert = (adv_pert_cnt * 100) / total_cnt
    fr = (fool_cnt * 100) / total_cnt
    adavops_fr = (advop_fr_cnt * 100) / total_cnt
    wirter.add_scalar("test/acc_clean", acc_clean, epoch + i)
    wirter.add_scalar("test/acc_adv", acc_adv, epoch + i)
    wirter.add_scalar("test/acc_advops", acc_adv_ops, epoch + i)
    wirter.add_scalar("test/acc_adv_pert", acc_adv_pert, epoch + i)
    wirter.add_scalar("test/fr", fr, epoch + i)
    wirter.add_scalar("test/acc_advops_fr", adavops_fr, epoch + i)

    print("【Test】: epoch %d:, clean_acc: %.3f, adv_acc: %.3f, fooling_rate: %.3f, acc_ops: %.3f\n" % (epoch,
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
    parser.add_argument("--loss_type", type=str, default='cw', help="cw,  ce")
    parser.add_argument("--lr_g", type=float, default=0.0002)
    parser.add_argument("--lr_d", type=float, default=0.0002)
    parser.add_argument("--beta1", type=float, default=0.5)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--input_size", type=int, default=224)
    parser.add_argument("--num_filter", type=int, default=64)
    parser.add_argument("--lambda", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=50)
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--num_workers", type=int, default=8)

    args = parser.parse_args()
    return args

def get_model_by_name(model_name):
    normalize = Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225])
    if "resnet50" in model_name.lower():
        model = models.resnet50(pretrained=True)
    elif "resnet34" in model_name.lower():
        model = models.resnet34(pretrained=True)
    elif "resnet101" in model_name.lower():
        model = models.resnet101(pretrained=True)
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

    model = torch.nn.Sequential(normalize, model)
    model.eval()
    model.to(device)
    return model

if __name__ == "__main__":
    args = get_args()

    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(args.input_size),
        transforms.ToTensor(),
    ])

    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(args.input_size),
        transforms.ToTensor(),
    ])
    traindir = os.path.join(args.data_dir,'ImageNet10k')
    valdir = os.path.join(args.data_dir, 'val')
    # dataset
    train_data = ImageFolder(root=traindir, transform=train_transform)
    test_data = ImageFolder(root=valdir, transform=test_transform)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                                               pin_memory=True, num_workers=args.num_workers, drop_last=True)
    validation_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, num_workers=args.num_workers)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    g = OpsAdvGenerator(3, args.num_filter, 3)
    d = Discriminator(3, args.num_filter, 1)
    g = g.to(device)
    # d = d.to(device)
    g.normal_weight_init(mean=0.0, std=0.02)
    d.normal_weight_init(mean=0.0, std=0.02)

    model_name_list = ["resnet50", "vgg16", "densenet121", "resnext", "wideresnet", "mnasnet", "squeezenet"]

    model = get_model_by_name(args.model_name)

    g_optimizer = torch.optim.RMSprop(g.parameters(), lr=args.lr_g, weight_decay=0.99, momentum=0.01)
    CosineLR = torch.optim.lr_scheduler.CosineAnnealingLR(g_optimizer, T_max=200, eta_min=0)
    
    # Loss functions
    bce_loss = torch.nn.BCELoss()
    ce_loss = torch.nn.CrossEntropyLoss()
    mse_loss = torch.nn.MSELoss()
    l1_loss = torch.nn.L1Loss()
    kl_loss = torch.nn.KLDivLoss()

    eps_nomial = 10.0
    eps = eps_nomial/255

    
    time_str = time.strftime("%m-%d-%H-%M", time.localtime())
    save_dir = f"saved_images/EXP_NAME_{time_str}_eps{int(eps_nomial)}"

    ckpt_save_dir = f"{args.ckpt_path}/EXP_NAME_{time_str}_eps{int(eps_nomial)}"
    if not os.path.exists(ckpt_save_dir):
        os.makedirs(ckpt_save_dir)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)


    logit_type = ["custom_loss"]
    
    time_str = time.strftime("%m-%d-%H-%M", time.localtime())
    wirter = SummaryWriter(f"runs/EXP_ImageNet_advopsloss_{time_str}_{args.model_name}_{args.loss_type}")

    print(f"Model: {args.model_name}_{args.loss_type}")
    # record metrics
    best_acc_adv = np.inf
    best_fooling_rate = 0
    best_acc_advops = 0
    for epoch in range(1, args.num_epochs+1):
        train_g(epoch, g)
        if epoch % 5 == 0:
            acc, acc_adv, fooling_rate, acc_advops = test(epoch, g)
            if acc_advops > 80 and best_acc_advops < acc_advops:
                best_acc_advops = acc_advops
                netG_file_name = f'{ckpt_save_dir}/Best_advops_acc_PixelShuffle_Generator_for_{args.model_name}_{args.loss_type}_epoch_{str(epoch)}_{acc_advops}.pth'
                torch.save(g.state_dict(), netG_file_name)
            if best_fooling_rate > 80 and best_fooling_rate < fooling_rate:
                best_fooling_rate = fooling_rate
                netG_file_name = f'{ckpt_save_dir}/Best_fooling_rate_PixelShuffle_Generator_for_{args.model_name}_{args.loss_type}_epoch{str(epoch)}.pth'
                torch.save(g.state_dict(), netG_file_name)

            if best_acc_adv < 30 and best_acc_adv > acc_adv:
                best_acc_adv = acc_adv
                netG_file_name = f'{ckpt_save_dir}/Best_adv_acc_PixelShuffle_Generator_for_{args.model_name}_{args.loss_type}_epoch{str(epoch)}.pth'
                torch.save(g.state_dict(), netG_file_name)
        CosineLR.step()
        if epoch % (20+1) == 0:
            netG_file_name = f'{ckpt_save_dir}/PixelShuffle_Generator_for_{args.model_name}_{args.loss_type}_epoch{str(epoch)}_{acc_advops}.pth'
            torch.save(g.state_dict(), netG_file_name)

    netG_file_name = f'{ckpt_save_dir}/Lastest_PixelShuffle_Generator_for_{args.model_name}_{args.loss_type}_epoch{str(epoch)}.pth'
    torch.save(g.state_dict(), netG_file_name)
    wirter.close()
    print("Finished!!")