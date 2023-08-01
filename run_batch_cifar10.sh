# Fixed Params
TARGET_NETS="resnet50 preactresnet18 wideresnet densenet121 vgg16 vgg19"
for target_net in $TARGET_NETS; do
    python3 main_cifar10.py \
      --model_name $target_net
done
