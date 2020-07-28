import os
import sys
import time
import glob
import logging
import argparse
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.utils.data.distributed
import torch.backends.cudnn as cudnn
from nvidia.dali.plugin.pytorch import DALIClassificationIterator
from apex.parallel import DistributedDataParallel as DDP
from apex import amp, parallel

from utils  import AverageMeter, accuracy, set_seed, EMA
from utils  import create_exp_dir, save_checkpoint, get_params
from losses import CrossEntropyLabelSmooth
from datasets import ImageList, pil_loader, cv2_loader
from datasets import get_train_transform, get_val_transform
from datasets import HybridTrainPipe, HybridValPipe
from networks import MobileNetV3_Large, MobileNetV3_Small
from lr_scheduler import LambdaLRWithMin


parser = argparse.ArgumentParser(
	description="Basic Pytorch ImageNet Example. There is no tricks such as mixup/autoaug/dropblock/droppath etc.",
	formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# various paths
parser.add_argument('--train_root', type=str, required=True, help='root path to training images')
parser.add_argument('--train_list', type=str, required=True, help='training image list')
parser.add_argument('--val_root', type=str, required=True, help='root path to validating images')
parser.add_argument('--val_list', type=str, required=True, help='validating image list')
parser.add_argument('--save', type=str, default='./checkpoints/', help='model and log saving path')
parser.add_argument('--snapshot', type=str, default='', help='checkpoint for reset')

# training hyper-parameters
parser.add_argument('--print_freq', type=int, default=100, help='print frequency')
parser.add_argument('--workers', type=int, default=8, help='number of workers to load dataset (global)')
parser.add_argument('--epochs', type=int, default=250, help='number of total training epochs')
parser.add_argument('--warmup_epochs', type=int, default=5, help='number of warmup epochs')
parser.add_argument('--batch_size', type=int, default=512, help='batch size (global)')
parser.add_argument('--lr', type=float, default=0.2, help='initial learning rate')
parser.add_argument('--lr_min', type=float, default=0.0, help='minimum learning rate')
parser.add_argument('--lr_scheduler', type=str, default='cosine_epoch', help='type of lr scheduler',
					choices=['linear_epoch', 'linear_batch', 'cosine_epoch', 'cosine_batch', 'step_epoch', 'step_batch'])
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-5, help='weight decay (wd)')
parser.add_argument('--no_wd_bias_bn', action='store_true', default=False, help='whether to remove wd on bias and bn')
parser.add_argument('--model', type=str, default='MobileNetV3_Large', help='type of model',
					choices=['MobileNetV3_Large', 'MobileNetV3_Small'])
parser.add_argument('--num_classes', type=int, default=1000, help='class number of training set')
parser.add_argument('--dropout_rate', type=float, default=0.0, help='dropout rate')
parser.add_argument('--zero_init_last_bn', action='store_true', default=False, help='zero initialize the last bn in each block')
parser.add_argument('--label_smooth', type=float, default=0.1, help='label smoothing')
parser.add_argument('--trans_mode', type=str, default='tv', help='mode of image transformation (tv/dali)')
parser.add_argument('--color_jitter', action='store_true', default=False, help='apply color augmentation or not')
parser.add_argument('--dali_cpu', action='store_true', default=False, help='runs CPU based DALI pipeline')
parser.add_argument('--ema_decay', type=float, default=0.0, help='whether to use EMA')

# amp and DDP hyper-parameters
parser.add_argument('--local_rank', type=int, default=0)
parser.add_argument('--sync_bn', action='store_true', help='enabling apex sync BN')
parser.add_argument('--opt_level', type=str, default=None)
parser.add_argument('--keep_batchnorm_fp32', type=str, default=None)
parser.add_argument('--loss_scale', type=str, default=None)
parser.add_argument('--channels_last', type=str, default='False')

# others
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--note', type=str, default='try', help='note for this run')


args, unparsed = parser.parse_known_args()
args.channels_last = eval(args.channels_last)

args.save = os.path.join(args.save, '{}-{}'.format(time.strftime("%Y%m%d-%H%M%S"), args.note))
if args.local_rank == 0:
	create_exp_dir(args.save, scripts_to_save=glob.glob('*.py')+glob.glob('*.sh'))

	log_format = '%(asctime)s %(message)s'
	logging.basicConfig(stream=sys.stdout, level=logging.INFO,
		format=log_format, datefmt='%m/%d %I:%M:%S %p')
	fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
	fh.setFormatter(logging.Formatter(log_format))
	logging.getLogger().addHandler(fh)

if hasattr(torch, 'channels_last') and  hasattr(torch, 'contiguous_format'):
	if args.channels_last:
		memory_format = torch.channels_last
	else:
		memory_format = torch.contiguous_format
else:
	memory_format = None


def main():
	set_seed(args.seed)
	cudnn.enabled=True
	cudnn.benchmark = True
	args.distributed = False
	if 'WORLD_SIZE' in os.environ:
		args.distributed = int(os.environ['WORLD_SIZE']) > 1
	args.gpu = 0
	args.world_size = 1
	if args.distributed:
		set_seed(args.local_rank)
		args.gpu = args.local_rank
		torch.cuda.set_device(args.gpu)
		torch.distributed.init_process_group(backend='nccl', init_method='env://')
		args.world_size = torch.distributed.get_world_size()
	if args.local_rank == 0:
		logging.info("args = {}".format(args))
		logging.info("unparsed_args = {}".format(unparsed))
		logging.info("distributed = {}".format(args.distributed))
		logging.info("sync_bn = {}".format(args.sync_bn))
		logging.info("opt_level = {}".format(args.opt_level))
		logging.info("keep_batchnorm_fp32 = {}".format(args.keep_batchnorm_fp32))
		logging.info("loss_scale = {}".format(args.loss_scale))
		logging.info("CUDNN VERSION: {}".format(torch.backends.cudnn.version()))

	# create model
	if args.model == 'MobileNetV3_Large':
		model = MobileNetV3_Large(args.num_classes, args.dropout_rate, args.zero_init_last_bn)
	elif args.model == 'MobileNetV3_Small':
		model = MobileNetV3_Small(args.num_classes, args.dropout_rate, args.zero_init_last_bn)
	else:
		raise Exception('invalid type of model')
	if args.sync_bn:
		if args.local_rank == 0: logging.info("using apex synced BN")
		model = parallel.convert_syncbn_model(model)
	model = model.cuda().to(memory_format=memory_format) if memory_format is not None else model.cuda()

	# define criterion and optimizer
	if args.label_smooth > 0.0:
		criterion = CrossEntropyLabelSmooth(args.num_classes, args.label_smooth)
	else:
		criterion = nn.CrossEntropyLoss()
	criterion = criterion.cuda()

	params = get_params(model) if args.no_wd_bias_bn else model.parameters()
	optimizer = torch.optim.SGD(params, args.lr,
								momentum=args.momentum,
								weight_decay=args.weight_decay)
	# Initialize Amp
	if args.opt_level is not None:
		model, optimizer = amp.initialize(model, optimizer,
										  opt_level=args.opt_level,
										  keep_batchnorm_fp32=args.keep_batchnorm_fp32,
										  loss_scale=args.loss_scale)

	# For distributed training, wrap the model with apex.parallel.DistributedDataParallel.
	# This must be done AFTER the call to amp.initialize.
	if args.distributed:
		# By default, apex.parallel.DistributedDataParallel overlaps communication with
		# computation in the backward pass.
		# delay_allreduce delays all communication to the end of the backward pass.
		model = DDP(model, delay_allreduce=True)
	else:
		model = nn.DataParallel(model)

	# exponential moving average
	if args.ema_decay > 0.0:
		ema = EMA(model, args.ema_decay)
		ema.register()
	else:
		ema = None

	# define transform and initialize dataloader
	batch_size = args.batch_size // args.world_size
	workers    = args.workers    // args.world_size
	if args.trans_mode == 'tv':
		train_transform = get_train_transform(args.color_jitter)
		val_transform   = get_val_transform()
		train_dataset   = ImageList(root=args.train_root, 
									list_path=args.train_list, 
									transform=train_transform)
		val_dataset     = ImageList(root=args.val_root, 
									list_path=args.val_list, 
									transform=val_transform)
		train_sampler = None
		val_sampler   = None
		if args.distributed:
			train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True)
			val_sampler   = torch.utils.data.distributed.DistributedSampler(val_dataset,  shuffle=False)
		train_loader = torch.utils.data.DataLoader(
							train_dataset, batch_size=batch_size, num_workers=workers,
							pin_memory=True, sampler=train_sampler, shuffle=(train_sampler is None))
		val_loader   = torch.utils.data.DataLoader(
							val_dataset,   batch_size=batch_size, num_workers=workers, 
							pin_memory=True, sampler=val_sampler,   shuffle=False)
		args.batches_per_epoch = len(train_loader)
	elif args.trans_mode == 'dali':
		pipe = HybridTrainPipe(batch_size=batch_size,
							   num_threads=workers,
							   device_id=args.local_rank,
							   root=args.train_root,
							   list_path=args.train_list,
							   crop=224,
							   shard_id=args.local_rank,
							   num_shards=args.world_size,
							   coji=args.color_jitter,
							   dali_cpu=args.dali_cpu)
		pipe.build()
		train_loader = DALIClassificationIterator(pipe, size=int(pipe.epoch_size("Reader")/args.world_size))
		args.batches_per_epoch = train_loader._size // train_loader.batch_size
		args.batches_per_epoch += (train_loader._size % train_loader.batch_size) != 0

		pipe =   HybridValPipe(batch_size=batch_size,
							   num_threads=workers,
							   device_id=args.local_rank,
							   root=args.val_root,
							   list_path=args.val_list,
							   size=256,
							   crop=224,
							   shard_id=args.local_rank,
							   num_shards=args.world_size,
							   dali_cpu=args.dali_cpu)
		pipe.build()
		val_loader   = DALIClassificationIterator(pipe, size=int(pipe.epoch_size("Reader")/args.world_size))
	else:
		raise Exception('invalid image transformation mode')

	# define learning rate scheduler
	scheduler = get_lr_scheduler(optimizer)

	best_acc_top1 = 0
	best_acc_top5 = 0
	start_epoch = 0

	# restart from snapshot
	if args.snapshot and os.path.isfile(args.snapshot):
		if args.local_rank == 0:
			logging.info('loading snapshot from {}'.format(args.snapshot))
		checkpoint = torch.load(args.snapshot, map_location=lambda storage,loc: storage.cuda(args.gpu))
		start_epoch = checkpoint['epoch']
		best_acc_top1 = checkpoint['best_acc_top1']
		best_acc_top5 = checkpoint['best_acc_top5']
		model.load_state_dict(checkpoint['model'])
		optimizer.load_state_dict(checkpoint['optimizer'])
		if checkpoint['ema'] is not None:
			ema.load_state_dict(checkpoint['ema'])
		if args.opt_level is not None:
			amp.load_state_dict(checkpoint['amp'])
		scheduler = get_lr_scheduler(optimizer)
		for epoch in range(start_epoch):
			if epoch < args.warmup_epochs:
				adjust_learning_rate(optimizer, scheduler, epoch, -1)
				warmup_lr = get_last_lr(optimizer)
				if args.local_rank == 0:
					logging.info('Epoch: %d, Warming-up lr: %e', epoch, warmup_lr)
			else:
				current_lr = get_last_lr(optimizer)
				if args.local_rank == 0:
					logging.info('Epoch: %d lr %e', epoch, current_lr)

			if epoch < args.warmup_epochs:
				for param_group in optimizer.param_groups:
					param_group['lr'] = args.lr
			else:
				if args.lr_scheduler in ['linear_epoch', 'cosine_epoch', 'step_epoch']:
					adjust_learning_rate(optimizer, scheduler, epoch, -1)
				if args.lr_scheduler in ['linear_batch', 'cosine_batch', 'step_batch']:
					for batch_idx in range(args.batches_per_epoch):
						adjust_learning_rate(optimizer, scheduler, epoch, batch_idx)

	# the main loop
	for epoch in range(start_epoch, args.epochs):
		if epoch < args.warmup_epochs:
			adjust_learning_rate(optimizer, scheduler, epoch, -1)
			warmup_lr = get_last_lr(optimizer)
			if args.local_rank == 0:
				logging.info('Epoch: %d, Warming-up lr: %e', epoch, warmup_lr)
		else:
			current_lr = get_last_lr(optimizer)
			if args.local_rank == 0:
				logging.info('Epoch: %d lr %e', epoch, current_lr)

		if args.distributed and args.trans_mode == 'tv':
			train_sampler.set_epoch(epoch)

		epoch_start = time.time()
		train_acc, train_obj = train(train_loader, model, ema, criterion, optimizer, scheduler, epoch)
		if args.local_rank == 0:
			logging.info('Train_acc: %f', train_acc)

		val_acc_top1, val_acc_top5, val_obj = validate(val_loader, model, ema, criterion)
		if args.local_rank == 0:
			logging.info('Val_acc_top1: %f', val_acc_top1)
			logging.info('Val_acc_top5: %f', val_acc_top5)
			logging.info('Epoch time: %ds.', time.time() - epoch_start)

		if args.local_rank == 0:
			is_best = False
			if val_acc_top1 > best_acc_top1:
				best_acc_top1 = val_acc_top1
				best_acc_top5 = val_acc_top5
				is_best = True
			save_checkpoint({
				'epoch': epoch + 1,
				'model': model.state_dict(),
				'ema': ema.state_dict() if ema is not None else None,
				'best_acc_top1': best_acc_top1,
				'best_acc_top5': best_acc_top5,
				'optimizer' : optimizer.state_dict(),
				'amp': amp.state_dict() if args.opt_level is not None else None,
				}, is_best, args.save)

		if epoch < args.warmup_epochs:
			for param_group in optimizer.param_groups:
				param_group['lr'] = args.lr
		else:
			adjust_learning_rate(optimizer, scheduler, epoch, -1)

		if args.trans_mode == 'dali':
			train_loader.reset()
			val_loader.reset()


def train(train_loader, model, ema, criterion, optimizer, scheduler, epoch):
	objs = AverageMeter()
	top1 = AverageMeter()
	top5 = AverageMeter()
	batch_time = AverageMeter()
	data_time  = AverageMeter()
	model.train()

	end = time.time()
	for batch_idx, data in enumerate(train_loader):
		data_time.update(time.time() - end)
		if args.trans_mode == 'tv':
			x = data[0].cuda(non_blocking=True)
			target = data[1].cuda(non_blocking=True)
		elif args.trans_mode == 'dali':
			x = data[0]['data'].cuda(non_blocking=True)
			target = data[0]['label'].squeeze().cuda(non_blocking=True).long()

		# forward
		batch_start = time.time()
		logits = model(x)
		loss = criterion(logits, target)

		# backward
		optimizer.zero_grad()
		if args.opt_level is not None:
			with amp.scale_loss(loss, optimizer) as scaled_loss:
				scaled_loss.backward()
		else:
			loss.backward()
		optimizer.step()
		if ema is not None: ema.update()
		batch_time.update(time.time() - batch_start)

		if batch_idx % args.print_freq == 0:
			# For better performance, don't accumulate these metrics every iteration,
			# since they may incur an allreduce and some host<->device syncs.
			prec1, prec5 = accuracy(logits, target, topk=(1, 5))
			if args.distributed:
				reduced_loss = reduce_tensor(loss.data)
				prec1 = reduce_tensor(prec1)
				prec5 = reduce_tensor(prec5)
			else:
				reduced_loss = loss.data
			objs.update(reduced_loss.item(), x.size(0))
			top1.update(prec1.item(), x.size(0))
			top5.update(prec5.item(), x.size(0))
			torch.cuda.synchronize()

			duration = 0 if batch_idx == 0 else time.time() - duration_start
			duration_start = time.time()
			if args.local_rank == 0:
				logging.info('TRAIN Step: %03d Objs: %e R1: %f R5: %f Duration: %ds BTime: %.3fs DTime: %.4fs', 
									batch_idx, objs.avg, top1.avg, top5.avg, duration, batch_time.avg, data_time.avg)
		
		adjust_learning_rate(optimizer, scheduler, epoch, batch_idx)
		end = time.time()

	return top1.avg, objs.avg


def validate(val_loader, model, ema, criterion):
	objs = AverageMeter()
	top1 = AverageMeter()
	top5 = AverageMeter()

	# if ema is not None: ema.apply()
	model.eval()

	for batch_idx, data in enumerate(val_loader):
		if args.trans_mode == 'tv':
			x = data[0].cuda(non_blocking=True)
			target = data[1].cuda(non_blocking=True)
		elif args.trans_mode == 'dali':
			x = data[0]['data'].cuda(non_blocking=True)
			target = data[0]['label'].squeeze().cuda(non_blocking=True).long()

		with torch.no_grad():
			logits = model(x)
			loss = criterion(logits, target)

		prec1, prec5 = accuracy(logits, target, topk=(1, 5))
		if args.distributed:
			reduced_loss = reduce_tensor(loss.data)
			prec1 = reduce_tensor(prec1)
			prec5 = reduce_tensor(prec5)
		else:
			reduced_loss = loss.data
		objs.update(reduced_loss.item(), x.size(0))
		top1.update(prec1.item(), x.size(0))
		top5.update(prec5.item(), x.size(0))

		if args.local_rank == 0 and batch_idx % args.print_freq == 0:
			duration = 0 if batch_idx == 0 else time.time() - duration_start
			duration_start = time.time()
			logging.info('VALIDATE Step: %03d Objs: %e R1: %f R5: %f Duration: %ds', batch_idx, objs.avg, top1.avg, top5.avg, duration)

		# if ema is not None: ema.restore()

	return top1.avg, top5.avg, objs.avg


def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= args.world_size
    return rt


def get_lr_scheduler(optimizer):
	if args.lr_scheduler == 'linear_epoch':
		total_steps = args.epochs - args.warmup_epochs
		lambda_func = lambda step: max(1.0-step/float(total_steps), 0)
		scheduler = LambdaLRWithMin(optimizer, lambda_func, args.lr_min)
	elif args.lr_scheduler == 'linear_batch':
		total_steps = (args.epochs - args.warmup_epochs) * args.batches_per_epoch
		lambda_func = lambda step: max(1.0-step/float(total_steps), 0)
		scheduler = LambdaLRWithMin(optimizer, lambda_func, args.lr_min)
	elif args.lr_scheduler == 'cosine_epoch':
		total_steps = args.epochs - args.warmup_epochs
		scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(total_steps), args.lr_min)
	elif args.lr_scheduler == 'cosine_batch':
		total_steps = (args.epochs - args.warmup_epochs) * args.batches_per_epoch
		scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(total_steps), args.lr_min)
	elif args.lr_scheduler == 'step_epoch':
		assert args.lr_min > 0.0, 'the minimum lr must be larger than 0 for "step" lr_scheduler'
		total_steps = args.epochs - args.warmup_epochs
		gamma = (args.lr_min / args.lr) ** (1.0 / total_steps)
		scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma)
	elif args.lr_scheduler == 'step_batch':
		assert args.lr_min > 0.0, 'the minimum lr must be larger than 0 for "step" lr_scheduler'
		total_steps = (args.epochs - args.warmup_epochs) * args.batches_per_epoch
		gamma = (args.lr_min / args.lr) ** (1.0 / total_steps)
		scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma)
	else:
		raise Exception('invalid type fo lr scheduler')

	return scheduler


def get_last_lr(optimizer):
	last_lrs = [param_group['lr'] for param_group in optimizer.param_groups]
	return last_lrs[0]


def adjust_learning_rate(optimizer, scheduler, epoch, batch_idx):
	'''
	batch_idx = -1: adjusts lr per epoch
	batch_idx >= 0: adjusts lr per batch
	'''
	if args.lr_scheduler in ['linear_epoch', 'cosine_epoch', 'step_epoch']:
		if epoch < args.warmup_epochs:
			if batch_idx == -1:
				warmup_lr = float(epoch + 1) / (args.warmup_epochs + 1) * args.lr
				for param_group in optimizer.param_groups:
					param_group['lr'] = warmup_lr
		else:
			if batch_idx == -1:
				scheduler.step()

	if args.lr_scheduler in ['linear_batch', 'cosine_batch', 'step_batch']:
		if epoch < args.warmup_epochs:
			batch_idx = epoch * args.batches_per_epoch + batch_idx
			total_batches = args.warmup_epochs * args.batches_per_epoch
			warmup_lr = float(batch_idx + 2) / (total_batches + 1) * args.lr
			for param_group in optimizer.param_groups:
				param_group['lr'] = warmup_lr
		else:
			if batch_idx >= 0:
				scheduler.step()


if __name__ == '__main__':
	main()
