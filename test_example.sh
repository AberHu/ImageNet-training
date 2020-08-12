CUDA_VISIBLE_DEVICES=0 python -u test.py \
				--val_root "Your ImageNet Val Set Path" \
				--val_list "ImageNet Val List" \
				--weights "Pretrained Weights" \
				--model 'MobileNetV3_Large' \
				--trans_mode 'tv' \
				--ema
