import torch
import torch.nn as nn
import numpy as np
import sys
sys.path.append('..')
from networks import HSwish, HSigmoid, Swish, Sigmoid

def compute_madd(module, inp, out):
	if isinstance(module, (nn.ReLU, nn.ReLU6, nn.LeakyReLU, nn.PReLU)):
		return compute_ReLU_madd(module, inp, out)
	elif isinstance(module, nn.ELU):
		return compute_ELU_madd(module, inp, out)
	elif isinstance(module, Sigmoid):
		return compute_Sigmoid_madd(module, inp, out)
	elif isinstance(module, HSigmoid):
		return compute_HSigmoid_madd(module, inp, out)
	elif isinstance(module, Swish):
		return compute_Swish_madd(module, inp, out)
	elif isinstance(module, HSwish):
		return compute_HSwish_madd(module, inp, out)
	elif isinstance(module, nn.Conv2d):
		return compute_Conv2d_madd(module, inp, out)
	elif isinstance(module, nn.ConvTranspose2d):
		return compute_ConvTranspose2d_madd(module, inp, out)
	elif isinstance(module, nn.BatchNorm2d):
		return compute_BatchNorm2d_madd(module, inp, out)
	elif isinstance(module, nn.Linear):
		return compute_Linear_madd(module, inp, out)
	elif isinstance(module, nn.MaxPool2d):
		return compute_MaxPool2d_madd(module, inp, out)
	elif isinstance(module, nn.AdaptiveMaxPool2d):
		return compute_AdaptiveMaxPool2d_madd(module, inp, out)
	elif isinstance(module, nn.AvgPool2d):
		return compute_AvgPool2d_madd(module, inp, out)
	elif isinstance(module, nn.AdaptiveAvgPool2d):
		return compute_AdaptiveAvgPool2d_madd(module, inp, out)
	else:
		print("[MAdd]: {} is not supported!".format(type(module).__name__))
		return 0


def compute_ReLU_madd(module, inp, out):
	assert isinstance(module, (nn.ReLU, nn.ReLU6))

	count = 1
	for i in inp.size()[1:]:
		count *= i

	return count


def compute_ELU_madd(module, inp, out):
	assert isinstance(module, nn.ELU)

	count = 1
	for i in inp.size()[1:]:
		count *= i
	total_mul = count + count
	total_add = count

	return total_mul + total_add


def compute_Sigmoid_madd(module, inp, out):
	assert isinstance(module, Sigmoid)

	count = 1
	for i in inp.size()[1:]:
		count *= i
	total_mul = count + count + count
	total_add = count

	return total_mul + total_add


def compute_HSigmoid_madd(module, inp, out):
	assert isinstance(module, HSigmoid)

	count = 1
	for i in inp.size()[1:]:
		count *= i
	total_mul = count + (count)
	total_add = count

	return total_mul + total_add


def compute_Swish_madd(module, inp, out):
	assert isinstance(module, Swish)

	count = 1
	for i in inp.size()[1:]:
		count *= i
	total_mul = count + (count + count + count)
	total_add = 0 + (count)

	return total_mul + total_add


def compute_HSwish_madd(module, inp, out):
	assert isinstance(module, HSwish)

	count = 1
	for i in inp.size()[1:]:
		count *= i
	total_mul = count + (count + count)
	total_add = 0 + (count)

	return total_mul + total_add


def compute_Conv2d_madd(module, inp, out):
	assert isinstance(module, nn.Conv2d)
	assert len(inp.size()) == 4 and len(inp.size()) == len(out.size())

	in_c = inp.size()[1]
	k_h, k_w = module.kernel_size
	out_c, out_h, out_w = out.size()[1:]
	groups = module.groups

	# ops per output element
	kernel_mul = k_h * k_w * (in_c // groups)
	kernel_add = kernel_mul - 1 + (0 if module.bias is None else 1)

	kernel_mul_group = kernel_mul * out_h * out_w * (out_c // groups)
	kernel_add_group = kernel_add * out_h * out_w * (out_c // groups)

	total_mul = kernel_mul_group * groups
	total_add = kernel_add_group * groups

	return total_mul + total_add


def compute_ConvTranspose2d_madd(module, inp, out):
	assert isinstance(module, nn.ConvTranspose2d)
	assert len(inp.size()) == 4 and len(inp.size()) == len(out.size())

	in_c, in_h, in_w = inp.size()[1:]
	k_h, k_w = module.kernel_size
	out_c, out_h, out_w = out.size()[1:]
	groups = module.groups

	kernel_mul = k_h * k_w * (in_c // groups)
	kernel_add = kernel_mul - 1 + (0 if module.bias is None else 1)

	kernel_mul_group = kernel_mul * in_h * in_w * (out_c // groups)
	kernel_add_group = kernel_add * in_h * in_w * (out_c // groups)

	total_mul = kernel_mul_group * groups
	total_add = kernel_add_group * groups

	return total_mul + total_add


def compute_BatchNorm2d_madd(module, inp, out):
	assert isinstance(module, nn.BatchNorm2d)
	assert len(inp.size()) == 4 and len(inp.size()) == len(out.size())

	in_c, in_h, in_w = inp.size()[1:]

	# 1. sub mean
	# 2. div standard deviation
	# 3. mul alpha
	# 4. add beta
	return 4 * in_c * in_h * in_w


def compute_Linear_madd(module, inp, out):
	assert isinstance(module, nn.Linear)
	assert len(inp.size()) == 2 and len(out.size()) == 2

	num_in_features = inp.size()[1]
	num_out_features = out.size()[1]

	mul = num_in_features
	add = num_in_features - 1  + (0 if module.bias is None else 1)

	return num_out_features * (mul + add)


def compute_MaxPool2d_madd(module, inp, out):
	assert isinstance(module, nn.MaxPool2d)
	assert len(inp.size()) == 4 and len(inp.size()) == len(out.size())

	if isinstance(module.kernel_size, (tuple, list)):
		k_h, k_w = module.kernel_size
	else:
		k_h, k_w = module.kernel_size, module.kernel_size
	out_c, out_h, out_w = out.size()[1:]

	return (k_h * k_w - 1) * out_h * out_w * out_c


def compute_AdaptiveMaxPool2d_madd(module, inp, out):
	assert isinstance(module, nn.AdaptiveMaxPool2d)
	assert len(inp.size()) == 4 and len(inp.size()) == len(out.size())

	in_c, in_h, in_w = inp.size()[1:]
	out_c, out_h, out_w = out.size()[1:]
	k_h = int(round(in_h / out_h))
	k_w = int(round(in_w / out_w))

	return (k_h * k_w - 1) * out_h * out_w * out_c


def compute_AvgPool2d_madd(module, inp, out):
	assert isinstance(module, nn.AvgPool2d)
	assert len(inp.size()) == 4 and len(inp.size()) == len(out.size())

	if isinstance(module.kernel_size, (tuple, list)):
		k_h, k_w = module.kernel_size
	else:
		k_h, k_w = module.kernel_size, module.kernel_size
	out_c, out_h, out_w = out.size()[1:]

	kernel_add = k_h * k_w - 1
	kernel_avg = 1

	return (kernel_add + kernel_avg) * out_h * out_w * out_c


def compute_AdaptiveAvgPool2d_madd(module, inp, out):
	assert isinstance(module, nn.AdaptiveAvgPool2d)
	assert len(inp.size()) == 4 and len(inp.size()) == len(out.size())

	in_c, in_h, in_w = inp.size()[1:]
	out_c, out_h, out_w = out.size()[1:]
	k_h = int(round(in_h / out_h))
	k_w = int(round(in_w / out_w))

	kernel_add = k_h * k_w - 1
	kernel_avg = 1

	return (kernel_add + kernel_avg) * out_h * out_w * out_c
