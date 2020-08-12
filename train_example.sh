CUDA_VISIBLE_DEVICES=0,1 python -u -m torch.distributed.launch --nproc_per_node=2 train.py \
	--train_root "Your ImageNet Train Set Path" \
	--val_root "Your ImageNet Val Set Path" \
	--train_list "ImageNet Train List" \
	--val_list "ImageNet Val List" \
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
	--note 'try'


