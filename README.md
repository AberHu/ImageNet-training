# ImageNet-training

Pytorch ImageNet training codes with various tricks, lr schedulers, distributed training, mixed precision training, DALI dataloader etc.

## Train
```
CUDA_VISIBLE_DEVICES=0 python -u train.py --train_root /path/to/imagenet/train_set --val_root /path/to/imagenet/val_set --train_list /path/to/imagenet/train_list --val_list /path/to/imagenet/val_list
```

Please refer to [train_example.sh](https://github.com/AberHu/ImageNet-training/train_example.py) for more details.

## Test
```
CUDA_VISIBLE_DEVICES=0 python -u test.py --val_root /path/to/imagenet/val_set --val_list /path/to/imagenet/val_list --weights /path/to/pretrained_weights
```

Please refer to [test_example.sh](https://github.com/AberHu/ImageNet-training/test_example.py) for more details.

## Model Profiling
Please refer to [profile_example.py](https://github.com/AberHu/ImageNet-training/profile_example.py) for more details.

## Tested on
Python == 3.7.6 <br>
pytorch == 1.5.1 <br>
torchvision == 0.6.1 <br>
nvidia.dali == 0.22.0 <br>
cuDNN == 7.6.5 <br>
apex from [this link](https://github.com/NVIDIA/apex.git)

## License
This repo is released under the MIT license. Please see the [LICENSE](https://github.com/AberHu/ImageNet-training/LICENSE) file for more information.
