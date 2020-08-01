import os
import argparse
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from nvidia.dali.plugin.pytorch import DALIClassificationIterator
from apex.parallel import DistributedDataParallel as DDP

from utils  import AverageMeter, accuracy
from datasets import ImageList, pil_loader, cv2_loader
from datasets import get_val_transform, HybridValPipe
from networks import MobileNetV3_Large, MobileNetV3_Small


parser = argparse.ArgumentParser(
	description="Basic Pytorch ImageNet Example. Testing.",
	formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# various paths
parser.add_argument('--val_root', type=str, required=True, help='root path to validating images')
parser.add_argument('--val_list', type=str, required=True, help='validating image list')
parser.add_argument('--weights', type=str,  required=True, help='checkpoint for testing')

# testing hyper-parameters
parser.add_argument('--workers', type=int, default=8, help='number of workers to load dataset (global)')
parser.add_argument('--batch_size', type=int, default=512, help='batch size (global)')
parser.add_argument('--model', type=str, default='MobileNetV3_Large', help='type of model',
					choices=['MobileNetV3_Large', 'MobileNetV3_Small'])
parser.add_argument('--num_classes', type=int, default=1000, help='class number of testing set')
parser.add_argument('--trans_mode', type=str, default='tv', help='mode of image transformation (tv/dali)')
parser.add_argument('--dali_cpu', action='store_true', default=False, help='runs CPU based DALI pipeline')
parser.add_argument('--ema', action='store_true', default=False, help='whether to use EMA')

# amp and DDP hyper-parameters
parser.add_argument('--local_rank', type=int, default=0)
parser.add_argument('--channels_last', type=str, default='False')


args, unparsed = parser.parse_known_args()
args.channels_last = eval(args.channels_last)

if hasattr(torch, 'channels_last') and  hasattr(torch, 'contiguous_format'):
	if args.channels_last:
		memory_format = torch.channels_last
	else:
		memory_format = torch.contiguous_format
else:
	memory_format = None


def main():
	cudnn.enabled=True
	cudnn.benchmark = True
	args.distributed = False
	if 'WORLD_SIZE' in os.environ:
		args.distributed = int(os.environ['WORLD_SIZE']) > 1
	args.gpu = 0
	args.world_size = 1
	if args.distributed:
		args.gpu = args.local_rank
		torch.cuda.set_device(args.gpu)
		torch.distributed.init_process_group(backend='nccl', init_method='env://')
		args.world_size = torch.distributed.get_world_size()

	# create model
	if args.model == 'MobileNetV3_Large':
		model = MobileNetV3_Large(args.num_classes, 0.0, False)
	elif args.model == 'MobileNetV3_Small':
		model = MobileNetV3_Small(args.num_classes, 0.0, False)
	else:
		raise Exception('invalid type of model')
	model = model.cuda().to(memory_format=memory_format) if memory_format is not None else model.cuda()

	# For distributed training, wrap the model with apex.parallel.DistributedDataParallel.
	# This must be done AFTER the call to amp.initialize.
	if args.distributed:
		# By default, apex.parallel.DistributedDataParallel overlaps communication with
		# computation in the backward pass.
		# delay_allreduce delays all communication to the end of the backward pass.
		model = DDP(model, delay_allreduce=True)
	else:
		model = nn.DataParallel(model)

	# define transform and initialize dataloader
	batch_size = args.batch_size // args.world_size
	workers    = args.workers    // args.world_size
	if args.trans_mode == 'tv':
		val_transform   = get_val_transform()
		val_dataset     = ImageList(root=args.val_root, 
									list_path=args.val_list, 
									transform=val_transform)
		val_sampler   = None
		if args.distributed:
			val_sampler   = torch.utils.data.distributed.DistributedSampler(val_dataset,  shuffle=False)
		val_loader   = torch.utils.data.DataLoader(
							val_dataset,   batch_size=batch_size, num_workers=workers, 
							pin_memory=True, sampler=val_sampler,   shuffle=False)
	elif args.trans_mode == 'dali':
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

	# restart from weights
	if args.weights and os.path.isfile(args.weights):
		if args.local_rank == 0:
			print('loading weights from {}'.format(args.weights))
		checkpoint = torch.load(args.weights, map_location=lambda storage,loc: storage.cuda(args.gpu))
		if args.ema:
			model.load_state_dict(checkpoint['ema'])
		else:
			model.load_state_dict(checkpoint['model'])

	val_acc_top1, val_acc_top5 = validate(val_loader, model)
	if args.local_rank == 0:
		print('Val_acc_top1: {:.2f}'.format(val_acc_top1))
		print('Val_acc_top5: {:.2f}'.format(val_acc_top5))


def validate(val_loader, model):
	top1 = AverageMeter()
	top5 = AverageMeter()

	model.eval()

	for data in tqdm(val_loader):
		if args.trans_mode == 'tv':
			x = data[0].cuda(non_blocking=True)
			target = data[1].cuda(non_blocking=True)
		elif args.trans_mode == 'dali':
			x = data[0]['data'].cuda(non_blocking=True)
			target = data[0]['label'].squeeze().cuda(non_blocking=True).long()

		with torch.no_grad():
			logits = model(x)

		prec1, prec5 = accuracy(logits, target, topk=(1, 5))
		if args.distributed:
			prec1 = reduce_tensor(prec1)
			prec5 = reduce_tensor(prec5)
		top1.update(prec1.item(), x.size(0))
		top5.update(prec5.item(), x.size(0))

	return top1.avg, top5.avg


def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= args.world_size
    return rt


if __name__ == '__main__':
	main()
