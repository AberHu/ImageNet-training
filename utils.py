import os
import shutil
import torch
import random
import numpy as np


def set_seed(seed):
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)


class AverageMeter(object):
	"""
	Computes and stores the average and current value
	Copied from: https://github.com/pytorch/examples/blob/master/imagenet/main.py
	"""
	def __init__(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
	""" Computes the precision@k for the specified values of k """
	maxk = max(topk)
	batch_size = target.size(0)

	_, pred = output.topk(maxk, 1, True, True)
	pred = pred.t()
	correct = pred.eq(target.view(1, -1).expand_as(pred))

	res = []
	for k in topk:
		correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
		res.append(correct_k.mul_(100.0 / batch_size))
	return res


def save_checkpoint(state, is_best, save):
	filename = os.path.join(save, 'checkpoint.pth.tar')
	torch.save(state, filename)
	if is_best:
		best_filename = os.path.join(save, 'model_best.pth.tar')
		shutil.copyfile(filename, best_filename)


def create_exp_dir(path, scripts_to_save=None):
	if not os.path.exists(path):
		os.makedirs(path)
	print('Experiment dir : {}'.format(path))

	if scripts_to_save is not None:
		os.makedirs(os.path.join(path, 'scripts'))
		for script in scripts_to_save:
			dst_file = os.path.join(path, 'scripts', os.path.basename(script))
			shutil.copyfile(script, dst_file)


def get_params(model):
	params_no_weight_decay = []
	params_weight_decay = []
	for pname, p in model.named_parameters():
		if pname.find('weight') >= 0 and len(p.size()) > 1:
			# print('include ', pname, p.size())
			params_weight_decay.append(p)
		else:
			# print('not include ', pname, p.size())
			params_no_weight_decay.append(p)
	assert len(list(model.parameters())) == len(params_weight_decay) + len(params_no_weight_decay)
	params = [dict(params=params_weight_decay), dict(params=params_no_weight_decay, weight_decay=0.)]
	return params


class EMA():
	def __init__(self, model, decay):
		self.model = model
		self.decay = decay
		self.shadow = {}

	def register(self):
		for name, state in self.model.state_dict().items():
			self.shadow[name] = state.clone()

	def update(self):
		for name, state in self.model.state_dict().items():
			assert name in self.shadow
			new_average = (1.0 - self.decay) * state + self.decay * self.shadow[name]
			self.shadow[name] = new_average.clone()
			del new_average

	def state_dict(self):
		return self.shadow

	def load_state_dict(self, state_dict):
		for name, state in state_dict.items():
			self.shadow[name] = state.clone()