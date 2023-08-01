import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.manifold import TSNE
from generator_cifar10 import Generator, Discriminator
from torchvision import transforms
from cifar10_models import *

from utils import *
import argparse
import time
import torchvision
import torchattacks
import seaborn as sns


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
    transform_test = transforms.Compose([
        # transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
    ])
    testset = torchvision.datasets.CIFAR10(root='cifar10_models/data', train=False,
                                           download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100,
                                             shuffle=False, num_workers=0)
    return testloader


def gen_features(validation_loader):
    features = []
    label_list = []
    for i, batch in enumerate(validation_loader):
        images, labels = batch
        images, labels = images.to(device), labels.to(device)
        targets_np = labels.data.cpu().numpy()


        adv_images = atk(images, labels)

        # output = model(images)
        output = model(adv_images)
        outputs_np = output.data.cpu().numpy()
        features.append(outputs_np)
        label_list.append(targets_np[:, np.newaxis])
        
    
    label_list = np.concatenate(label_list, axis=0)
    features = np.concatenate(features, axis=0).astype(np.float64)
    return features, label_list

def gen_feature_generator(validation_loader):
    features = []
    label_list = []

    for i, batch in enumerate(validation_loader):
        images, labels = batch
        images, labels = images.to(device), labels.to(device)
        # targets_np = labels.data.cpu().numpy()
        pert = g(images)
        adv_images = torch.clamp(images + pert, 0, 1)
        output = model(adv_images)
        pred = torch.argmax(output, dim=1)
        targets_np = pred.data.cpu().numpy()
        outputs_np = output.data.cpu().numpy()

        features.append(outputs_np)
        # label_list.append(targets_np)
        label_list.append(targets_np[:, np.newaxis])

    label_list = np.concatenate(label_list, axis=0)
    features = np.concatenate(features, axis=0).astype(np.float64)
    return features, label_list

def tsne_plot(save_dir, targets, outputs):
    print('generating t-SNE plot...')
    # tsne_output = bh_sne(outputs)
    tsne = TSNE(random_state=0)
    tsne_output = tsne.fit_transform(outputs)

    df = pd.DataFrame(tsne_output, columns=['x', 'y'])
    df['class'] = targets

    plt.rcParams['figure.figsize'] = 10, 10
    sns.scatterplot(
        x='x', y='y',
        hue='class',
        palette=sns.color_palette("hls", 10),
        data=df,
        marker='o',
        legend="full",
        alpha=0.5
    )

    plt.xticks([])
    plt.yticks([])
    plt.xlabel('')
    plt.ylabel('')

    plt.savefig(save_dir, bbox_inches='tight')
    print('done!')



if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    validation_loader = load_cifar10("")
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
    model_name = list(model_name_list.keys())[0]
    model = load_model(model_name, list(model_name_list.values())[0])
    
    save_dir = 'tsne_figures'
    baseline = False

    # features, targets = gen_features(validation_loader)
    # print(np.shape(features))
    # print(np.shape(targets))
    # save_name = os.path.join(save_dir, f'cifar10_tsne_{model_name}.png')
    # tsne_plot(save_name, targets, features)

    if baseline:
        # for attack in ["FGSM", "BIM", "PGD", "MIFGSM"]:
        for attack in ["FGSM"]:
            if attack == "PGD":
                atk = torchattacks.PGD(model, eps=10.0 / 255)
            elif attack == "FGSM":
                atk = torchattacks.FGSM(model, eps=10.0 / 255)
            elif attack == "BIM":
                atk = torchattacks.BIM(model, eps=10.0 / 255)
            elif attack == "MIFGSM":
                atk = torchattacks.MIFGSM(model, eps=10.0 / 255)
            elif attack == "CW":
                atk = torchattacks.CW(model)  # tensor(15.2147, device='cuda:0')
            features, targets = gen_features(validation_loader)
            save_name = os.path.join(save_dir,f'{attack}_tsne_{model_name}.png')
            tsne_plot(save_name, targets, features)
    else:
        ckpt_file = '/mnt/jfs/wangdonghua/pythonpro/AdvOps/checkpoints/EXP_NAME_10-24-12-17_eps10/CIFAR10_Best_advops_acc_Generator_for_preactresnet18_custom_loss.pth'
        g = Generator(3, 3)
        g.load_state_dict(torch.load(ckpt_file))
        g = g.to(device)
        g.eval()
        features, targets = gen_feature_generator(validation_loader)
        print(1111)
        save_name = os.path.join(save_dir,f'generator_tsne_{model_name}_predict_class.png')
        tsne_plot(save_name, targets, features)
