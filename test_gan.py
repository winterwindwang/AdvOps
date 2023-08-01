import numpy as np

from generator import OpsAdvGenerator, Discriminator
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
import os
from torch.autograd import Variable
import torch.nn.functional as F
import torch
from utils import *
import argparse
import torchvision


@torch.no_grad()
def test(generator, epoch):
    fooling_num = 0
    accuracy_op = 0
    accuracy_clean = 0
    accuracy_adv = 0
    n = 0
    for i, batch in enumerate(validation_loader):
        images, labels = batch
        images, labels = images.to(device), labels.to(device)

        # train the discriminator with fake data
        perturbation = generator(images)
        adv_image = torch.clamp(images + perturbation, 0, 1)

        # Generator
        f_adv = model(adv_image)
        f_perturbation = model(perturbation)
        f_clean = model(images)

        clean_pred = torch.argmax(f_clean, dim=1)
        adv_pred = torch.argmax(f_adv, dim=1)

        # f_adv = f_perturbation + f_clean
        adv_ops_pred = torch.argmax(f_clean + f_perturbation, dim=1)

        # transforms.ToPILImage()(adv_image[0]).show()
        # transforms.ToPILImage()(images[0]).show()
        # transforms.ToPILImage()(perturbation[0]).show()

        accuracy_op += torch.sum(adv_pred == adv_ops_pred).item()
        fooling_num += torch.sum(adv_pred != clean_pred).item()
        accuracy_clean += torch.sum(clean_pred == labels).item()
        accuracy_adv += torch.sum(adv_pred == labels).item()
        n += images.size(0)

    print("epoch %d:\n, clean_acc: %.3f,\
                \nadv_acc: %.3f, fooling_rate: %.3f, acc_ops: %.3f\n" % (epoch,
        round(accuracy_clean*100 / n, 2), round(accuracy_adv*100 / n, 2),
        round(fooling_num*100 / n, 2),round(accuracy_op*100 / n, 2)))

    return round(accuracy_clean*100 / n, 2), round(accuracy_adv*100 / n, 2), round(fooling_num*100 / n, 2), round(accuracy_op*100 / n, 2)

def get_args():
    parser = argparse.ArgumentParser(description="Args Container")
    parser.add_argument("--data_dir", type=str, default='/mnt/jfs/wangdonghua/dataset/ImageNet/')
    parser.add_argument("--ckpt_path", type=str, default='checkpoints/')
    parser.add_argument("--lr_g", type=float, default=0.0002)
    parser.add_argument("--lr_d", type=float, default=0.0002)
    parser.add_argument("--beta1", type=float, default=0.5)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--input_size", type=int, default=224)
    parser.add_argument("--num_filter", type=int, default=64)
    parser.add_argument("--lambda", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_epochs", type=int, default=100)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()

    test_transform = transforms.Compose([
        transforms.Resize(256),
        # transforms.Resize(299), # inception_v3
        transforms.CenterCrop(args.input_size),
        transforms.ToTensor(),
    ])
    valdir = os.path.join(args.data_dir, 'val')
    # dataset
    test_data = ImageFolder(root=valdir, transform=test_transform)
    validation_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    g = OpsAdvGenerator(3, args.num_filter, 3)
    g = g.to(device)

    g.load_state_dict(torch.load("checkpoints/Lastest_Generator_epoch_v1.pth"))
    g.eval()

    model = models.resnet50(pretrained=True)
    normalize = Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225])
    model = torch.nn.Sequential(normalize, model)
    model.eval()
    model.to(device)

    acc, acc_adv, fooling_rate, acc_advops = test(g,0)

    print("Finished!!")