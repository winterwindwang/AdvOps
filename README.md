[TOC]

# AdvOps: Decoupling Adversarial Examples

The official code for **AdvOps: Decoupling Adversarial Examples**.

We introduce the implementation of the **AdvOps** for ImageNet and CIFAR-10 as following.

we use the directory of the ImageNet as follows

--ImageNet

​		|--class_2

​				|-- xxx.png

​				|-- xxx.png

​		|--class_2

​				|-- xxx.png

​				|-- xxx.png

​		....

We adopt the `ImageFolder` provide by torch to load dataset.

## ImageNet 

### Train generator

We use the ImageNet pretrained model provided by the PyTorch as the target model. To train the generator, please run the follow command

```bash
python main.py --model_name resnet50 --data_dir "data path" --ckpt_path "path to save the model parameters"  --batch_size 50
```

another parameters can be seen in `main.py`. 

We also provide the trained model of the generator for ImageNet in [here](https://pan.baidu.com/s/1FFBVEPJ7IEbEL0RQXFsaiw ) with the fetch code `1111`, CIFAR10 in [here](https://pan.baidu.com/s/1jQ9EtVWjZDu6BE5zE_lZKw), with the fetch code `1111`.

We will introduce the evaluation process in the following part.



## CIFAR-10

As pytorch do not provide the CIFAR-10 pretraining model, therefore, we have to train the model on CIFAR-10 first, then to train the generator.

### Step 1: Train CIFAR-10 model

Using the following common to train all CIFAR-10 model.

```python
python train_cifar10_model.py --ckpt_path "path to save the model checkpoint"
```

We also provide our pretrained CIFAR10 model in [here](https://pan.baidu.com/s/1_-ticZbQiVLVchoW3GzwUQ) with fetch code `1111`.

## Step 2: Train generator for each model

Using the following common to train the generator for each model

```python
python main_cifar10.py --model_name resnet50 --data_dir "data path" --model_ckpt "path of trained CIFAR-10 model" --ckpt_path "path to save the model parameters"  --batch_size 200
```

## Evaluation

#### Evaluate the AdvOps

To evaluate the trained generator, using the following command for ImageNet and CIFAR10, respectively.

For ImageNet

```python
python baseline_methods_transfer_gan.py
```

For CIFAR-10

```python
python baseline_methods_transfer_gan_cifar10.py
```



### Evaluate the comparison methods

To evaluate the comparison method, e.g., FGSM, using the following command

**For ImageNet**

```python
python baseline_methods_transfer.py
```

**For CIFAR-10**

```python
python baseline_methods_transfer_cifar10.py
```

**Note that** please the model checkpoint and dataset in the corresponding directory before runing the code.
