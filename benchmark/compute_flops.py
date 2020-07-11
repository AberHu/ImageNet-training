import torch
import torch.nn as nn
import numpy as np
import sys
sys.path.append('..')
from networks import HSwish, HSigmoid, Swish, Sigmoid


def compute_flops(module, inp, out):
	if isinstance(module, (nn.ReLU, nn.ReLU6, nn.PReLU, nn.LeakyReLU)):
		return compute_ReLU_flops(module, inp, out), 'Activation'
	elif isinstance(module, nn.ELU):
		return compute_ELU_flops(module, inp, out), 'Activation'
	elif isinstance(module, Sigmoid):
		return compute_Sigmoid_flops(module, inp, out), 'Activation'
	elif isinstance(module, HSigmoid):
		return compute_HSigmoid_flops(module, inp, out), 'Activation'
	elif isinstance(module, Swish):
		return compute_Swish_flops(module, inp, out), 'Activation'
	elif isinstance(module, HSwish):
		return compute_HSwish_flops(module, inp, out), 'Activation'
	elif isinstance(module, nn.Conv2d):
		return compute_Conv2d_flops(module, inp, out), 'Conv2d'
	elif isinstance(module, nn.ConvTranspose2d):
		return compute_ConvTranspose2d_flops(module, inp, out), 'ConvTranspose2d'
	elif isinstance(module, nn.BatchNorm2d):
		return compute_BatchNorm2d_flops(module, inp, out), 'BatchNorm2d'
	elif isinstance(module, (nn.AvgPool2d, nn.MaxPool2d)):
		return compute_Pool2d_flops(module, inp, out), 'Pool2d'
	elif isinstance(module, (nn.AdaptiveAvgPool2d, nn.AdaptiveMaxPool2d)):
		return compute_AdaptivePool2d_flops(module, inp, out), 'Pool2d'
	elif isinstance(module, nn.Linear):
		return compute_Linear_flops(module, inp, out), 'Linear'
	else:
		print("[Flops]: {} is not supported!".format(type(module).__name__))
		return 0, -1
	pass


def compute_ReLU_flops(module, inp, out):
	assert isinstance(module, (nn.ReLU, nn.ReLU6, nn.PReLU, nn.LeakyReLU))

	batch_size = inp.size()[0]
	active_elements_count = batch_size

	for s in inp.size()[1:]:
		active_elements_count *= s

	return active_elements_count


def compute_ELU_flops(module, inp, out):
	assert isinstance(module, nn.ELU)

	batch_size = inp.size()[0]
	active_elements_count = batch_size

	for s in inp.size()[1:]:
		active_elements_count *= s
	active_elements_count *= 3

	return active_elements_count


def compute_Sigmoid_flops(module, inp, out):
	assert isinstance(module, Sigmoid)

	batch_size = inp.size()[0]
	active_elements_count = batch_size

	for s in inp.size()[1:]:
		active_elements_count *= s
	active_elements_count *= 4

	return active_elements_count


def compute_HSigmoid_flops(module, inp, out):
	assert isinstance(module, HSigmoid)

	batch_size = inp.size()[0]
	active_elements_count = batch_size

	for s in inp.size()[1:]:
		active_elements_count *= s
	active_elements_count *= (2 + 1)

	return active_elements_count


def compute_Swish_flops(module, inp, out):
	assert isinstance(module, Swish)

	batch_size = inp.size()[0]
	active_elements_count = batch_size

	for s in inp.size()[1:]:
		active_elements_count *= s
	active_elements_count *= (1 + 4)

	return active_elements_count


def compute_HSwish_flops(module, inp, out):
	assert isinstance(module, HSwish)

	batch_size = inp.size()[0]
	active_elements_count = batch_size

	for s in inp.size()[1:]:
		active_elements_count *= s
	active_elements_count *= (1 + 3)

	return active_elements_count


def compute_Conv2d_flops(module, inp, out):
	# Can have multiple inputs, getting the first one
	assert isinstance(module, nn.Conv2d)
	assert len(inp.size()) == 4 and len(inp.size()) == len(out.size())

	batch_size = inp.size()[0]
	in_c = inp.size()[1]
	k_h, k_w = module.kernel_size
	out_c, out_h, out_w = out.size()[1:]
	groups = module.groups

	conv_per_position_flops = k_h * k_w * in_c * out_c // groups
	active_elements_count = batch_size * out_h * out_w
	total_conv_flops = conv_per_position_flops * active_elements_count

	bias_flops = 0
	if module.bias is not None:
		bias_flops = out_c * active_elements_count

	total_flops = total_conv_flops + bias_flops
	return total_flops


def compute_ConvTranspose2d_flops(module, inp, out):
	# Can have multiple inputs, getting the first one
	assert isinstance(module, nn.ConvTranspose2d)
	assert len(inp.size()) == 4 and len(inp.size()) == len(out.size())

	batch_size = inp.size()[0]
	in_c = inp.size()[1]
	k_h, k_w = module.kernel_size
	out_c, out_h, out_w = out.size()[1:]
	groups = module.groups

	conv_per_position_flops = k_h * k_w * in_c * out_c // groups
	active_elements_count = batch_size * out_h * out_w
	total_conv_flops = conv_per_position_flops * active_elements_count

	bias_flops = 0
	if module.bias is not None:
		bias_flops = out_c * active_elements_count

	total_flops = total_conv_flops + bias_flops
	return total_flops


def compute_BatchNorm2d_flops(module, inp, out):
	assert isinstance(module, nn.BatchNorm2d)
	assert len(inp.size()) == 4 and len(inp.size()) == len(out.size())

	bn_flops = np.prod(inp.shape)
	if module.affine:
		bn_flops *= 2
	
	return bn_flops


def compute_Pool2d_flops(module, inp, out):
	assert isinstance(module, (nn.MaxPool2d, nn.AvgPool2d))
	assert len(inp.size()) == 4 and len(inp.size()) == len(out.size())

	if isinstance(module.kernel_size, (tuple, list)):
		k_h, k_w = module.kernel_size
	else:
		k_h, k_w = module.kernel_size, module.kernel_size
	out_c, out_h, out_w = out.size()[1:]
	batch_size = inp.size()[0]

	pool_flops = batch_size * out_c * out_h * out_w * k_h * k_w

	return pool_flops


def compute_AdaptivePool2d_flops(module, inp, out):
	assert isinstance(module, (nn.AdaptiveAvgPool2d, nn.AdaptiveMaxPool2d))
	assert len(inp.size()) == 4 and len(inp.size()) == len(out.size())

	inp_c, inp_h, inp_w = inp.size()[1:]
	out_c, out_h, out_w = out.size()[1:]
	k_h = int(round(inp_h / out_h))
	k_w = int(round(inp_w / out_w))
	batch_size = inp.size()[0]

	adaptive_pool_flops = batch_size * out_c * out_h * out_w * k_h * k_w

	return np.prod(inp.shape)


def compute_Linear_flops(module, inp, out):
	assert isinstance(module, nn.Linear)
	assert len(inp.size()) == 2 and len(out.size()) == 2

	batch_size = inp.size()[0]
	num_in_features = inp.size()[1]
	num_out_features = out.size()[1]

	total_fc_flops = batch_size * num_in_features * num_out_features

	bias_flops = 0
	if module.bias is not None:
		bias_flops = batch_size * num_out_features

	total_flops = total_fc_flops + bias_flops
	return total_flops

