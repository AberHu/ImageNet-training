CUDA_VISIBLE_DEVICES=1 python -u train.py \
				--train_root "/export/yibo.hu/dataset/ILSVRC2015/Data/CLS-LOC/train/" \
				--val_root "/export/yibo.hu/dataset/ILSVRC2015/Data/CLS-LOC/val/" \
				--train_list "/export/yibo.hu/dataset/ILSVRC2015/ILSVRC2015_train_cls.txt" \
				--val_list "/export/yibo.hu/dataset/ILSVRC2015/ILSVRC2015_val_cls.txt" \
				--save './checkpoints/' \
				--workers 16 \
				--epochs 250 \
				--warmup_epochs 5 \
				--batch_size 512 \
				--lr 0.2 \
				--lr_min 0.0 \
				--lr_scheduler 'cosine_epoch' \
				--momentum 0.9 \
				--weight_decay 3e-5 \
				--no_wd_bias_bn \
				--model 'MobileNetV3_Large' \
				--num_classes 1000 \
				--dropout_rate 0.2 \
				--label_smooth 0.1 \
				--trans_mode 'tv' \
				--color_jitter \
				--ema_decay 0.9999 \
				--opt_level 'O1' \
				--note 'MobileNetV3_Large-e250-we5-bs512-lr0.2-lrm0.0-cos_e-wd3e-5-no_wd-do0.2-ls0.1-tv-coji-ema-ampO1'


