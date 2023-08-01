# Fixed Params
TARGET_NETS="resnet50 vgg16 densenet121 resnext50 wideresnet mnasnet10 squeezenet"
for target_net in $TARGET_NETS; do
    python3 main.py \
      --model_name $target_net
done
