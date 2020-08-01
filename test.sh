CUDA_VISIBLE_DEVICES=2 python -u test.py \
				--val_root "/export/yibo.hu/dataset/ILSVRC2015/Data/CLS-LOC/val/" \
				--val_list "/export/yibo.hu/dataset/ILSVRC2015/ILSVRC2015_val_cls.txt" \
				--weights "" \
				--model 'MobileNetV3_Large' \
				--trans_mode 'tv' \
				--ema
