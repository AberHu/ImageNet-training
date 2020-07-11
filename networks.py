import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


class HSwish(nn.Module):
	def __init__(self, inplace=True):
		super(HSwish, self).__init__()
		self.inplace = inplace

	def forward(self, x):
		out = x * F.relu6(x + 3, inplace=self.inplace) / 6
		return out

class HSigmoid(nn.Module):
	def __init__(self, inplace=True):
		super(HSigmoid, self).__init__()
		self.inplace = inplace

	def forward(self, x):
		out = F.relu6(x + 3, inplace=self.inplace) / 6
		return out

class Swish(nn.Module):
	def __init__(self):
		super(Swish, self).__init__()

	def forward(self, x):
		out = x * F.sigmoid(x)
		return out

Sigmoid = nn.Sigmoid


hswish = HSwish
hsigmoid = HSigmoid
swish = Swish
sigmoid = Sigmoid
relu = nn.ReLU
relu6 = nn.ReLU6


class SEModule(nn.Module):
	def __init__(self, in_channels, reduction=4):
		super(SEModule, self).__init__()
		self.se = nn.Sequential(
			nn.AdaptiveAvgPool2d(1),
			nn.Conv2d(in_channels, in_channels//reduction, kernel_size=1, stride=1, padding=0, bias=True),
			nn.ReLU(inplace=True),
			nn.Conv2d(in_channels//reduction, in_channels, kernel_size=1, stride=1, padding=0, bias=True),
			hsigmoid(inplace=True)
		)

	def forward(self, x):
		return x * self.se(x)


class MBInvertedResBlock(nn.Module):
	def __init__(self, in_channels, mid_channels, out_channels, kernel_size=3, stride=1, act_func=relu, with_se=False):
		super(MBInvertedResBlock, self).__init__()
		self.has_residual = (in_channels == out_channels) and (stride == 1)
		self.se = SEModule(mid_channels) if with_se else None

		if mid_channels > in_channels:
			self.inverted_bottleneck = nn.Sequential(
					nn.Conv2d(in_channels, mid_channels, kernel_size=1, stride=1, padding=0, bias=False),
					nn.BatchNorm2d(mid_channels),
					act_func(inplace=True)
				)
		else:
			self.inverted_bottleneck = None
		self.depth_conv = nn.Sequential(
				nn.Conv2d(mid_channels, mid_channels, kernel_size=kernel_size, stride=stride, 
						  padding=kernel_size//2, groups=mid_channels, bias=False),
				nn.BatchNorm2d(mid_channels),
				act_func(inplace=True)
			)
		self.point_linear = nn.Sequential(
				nn.Conv2d(mid_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
				nn.BatchNorm2d(out_channels)
			)

	def forward(self, x):
		res = x

		if self.inverted_bottleneck is not None:
			out = self.inverted_bottleneck(x)
		else:
			out = x

		out = self.depth_conv(out)
		if self.se is not None:
			out = self.se(out)
		out = self.point_linear(out)

		if self.has_residual:
			out += res

		return out


class MobileNetV3_Large(nn.Module):
	def __init__(self, num_classes=1000, dropout_rate=0.0, zero_init_last_bn=False):
		super(MobileNetV3_Large, self).__init__()
		self.dropout_rate = dropout_rate
		self.zero_init_last_bn = zero_init_last_bn

		self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False)
		self.bn1   = nn.BatchNorm2d(16)
		self.hs1   = hswish(inplace=True)
		self.bneck = nn.Sequential(
				MBInvertedResBlock(16,  16,  16,  3, 1, relu,   False),
				MBInvertedResBlock(16,  64,  24,  3, 2, relu,   False),
				MBInvertedResBlock(24,  72,  24,  3, 1, relu,   False),
				MBInvertedResBlock(24,  72,  40,  5, 2, relu,   True),
				MBInvertedResBlock(40,  120, 40,  5, 1, relu,   True),
				MBInvertedResBlock(40,  120, 40,  5, 1, relu,   True),
				MBInvertedResBlock(40,  240, 80,  3, 2, hswish, False),
				MBInvertedResBlock(80,  200, 80,  3, 1, hswish, False),
				MBInvertedResBlock(80,  184, 80,  3, 1, hswish, False),
				MBInvertedResBlock(80,  184, 80,  3, 1, hswish, False),
				MBInvertedResBlock(80,  480, 112, 3, 1, hswish, True),
				MBInvertedResBlock(112, 672, 112, 3, 1, hswish, True),
				MBInvertedResBlock(112, 672, 160, 5, 2, hswish, True),
				MBInvertedResBlock(160, 960, 160, 5, 1, hswish, True),
				MBInvertedResBlock(160, 960, 160, 5, 1, hswish, True),
			)
		self.conv2   = nn.Conv2d(160, 960, kernel_size=1, stride=1, padding=0, bias=False)
		self.bn2     = nn.BatchNorm2d(960)
		self.hs2     = hswish(inplace=True)
		self.avgpool = nn.AdaptiveAvgPool2d(1)
		self.conv3   = nn.Conv2d(960, 1280, kernel_size=1, stride=1, padding=0, bias=True)
		self.hs3     = hswish()
		self.classifier = nn.Linear(1280, num_classes)

		self._initialization()
		# self._set_bn_param(0.1, 0.001)

	def forward(self, x):
		out = self.conv1(x)
		out = self.bn1(out)
		out = self.hs1(out)

		out = self.bneck(out)
		out = self.conv2(out)
		out = self.bn2(out)
		out = self.hs2(out)

		out = self.avgpool(out)
		out = self.conv3(out)
		out = self.hs3(out)
		out = out.view(out.size(0), -1)
		if self.dropout_rate > 0.0:
			out = F.dropout(out, p=self.dropout_rate, training=self.training)
		out = self.classifier(out)

		return out

	def _initialization(self):
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				init.kaiming_normal_(m.weight, mode='fan_out')
				if m.bias is not None:
					init.constant_(m.bias, 0)
			elif isinstance(m, nn.BatchNorm2d):
				init.constant_(m.weight, 1)
				init.constant_(m.bias, 0)
			elif isinstance(m, nn.Linear):
				init.normal_(m.weight, std=0.001)
				if m.bias is not None:
					init.constant_(m.bias, 0)

		if self.zero_init_last_bn:
			for mname, m in self.named_modules():
				if isinstance(m, MBInvertedResBlock):
					if m.has_residual:
						init.constant_(m.point_linear[1].weight, 0)

	# def _set_bn_param(self, bn_momentum, bn_eps):
	# 	for m in self.modules():
	# 		if isinstance(m, nn.BatchNorm2d):
	# 			m.momentum = bn_momentum
	# 			m.eps = bn_eps


class MobileNetV3_Small(nn.Module):
	def __init__(self, num_classes=1000, dropout_rate=0.0, zero_init_last_bn=False):
		super(MobileNetV3_Small, self).__init__()
		self.dropout_rate = dropout_rate
		self.zero_init_last_bn = zero_init_last_bn

		self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False)
		self.bn1   = nn.BatchNorm2d(16)
		self.hs1   = hswish(inplace=True)
		self.bneck = nn.Sequential(
				MBInvertedResBlock(16, 16,  16, 3, 2, relu,   True),
				MBInvertedResBlock(16, 72,  24, 3, 2, relu,   False),
				MBInvertedResBlock(24, 88,  24, 3, 1, relu,   False),
				MBInvertedResBlock(24, 96,  40, 5, 2, hswish, True),
				MBInvertedResBlock(40, 240, 40, 5, 1, hswish, True),
				MBInvertedResBlock(40, 240, 40, 5, 1, hswish, True),
				MBInvertedResBlock(40, 120, 48, 5, 1, hswish, True),
				MBInvertedResBlock(48, 144, 48, 5, 1, hswish, True),
				MBInvertedResBlock(48, 288, 96, 5, 2, hswish, True),
				MBInvertedResBlock(96, 576, 96, 5, 1, hswish, True),
				MBInvertedResBlock(96, 576, 96, 5, 1, hswish, True),
			)
		self.conv2   = nn.Conv2d(96, 576, kernel_size=1, stride=1, padding=0, bias=False)
		self.bn2     = nn.BatchNorm2d(576)
		self.hs2     = hswish(inplace=True)
		self.avgpool = nn.AdaptiveAvgPool2d(1)
		self.conv3   = nn.Conv2d(576, 1280, kernel_size=1, stride=1, padding=0, bias=True)
		self.hs3     = hswish()
		self.classifier = nn.Linear(1280, num_classes)

		self._initialization()
		# self._set_bn_param(0.1, 0.001)

	def forward(self, x):
		out = self.conv1(x)
		out = self.bn1(out)
		out = self.hs1(out)

		out = self.bneck(out)
		out = self.conv2(out)
		out = self.bn2(out)
		out = self.hs2(out)

		out = self.avgpool(out)
		out = self.conv3(out)
		out = self.hs3(out)
		out = out.view(out.size(0), -1)
		if self.dropout_rate > 0.0:
			out = F.dropout(out, p=self.dropout_rate, training=self.training)
		out = self.classifier(out)

		return out

	def _initialization(self):
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				init.kaiming_normal_(m.weight, mode='fan_out')
				if m.bias is not None:
					init.constant_(m.bias, 0)
			elif isinstance(m, nn.BatchNorm2d):
				init.constant_(m.weight, 1)
				init.constant_(m.bias, 0)
			elif isinstance(m, nn.Linear):
				init.normal_(m.weight, std=0.001)
				if m.bias is not None:
					init.constant_(m.bias, 0)

		if self.zero_init_last_bn:
			for mname, m in self.named_modules():
				if isinstance(m, MBInvertedResBlock):
					if m.has_residual:
						init.constant_(m.point_linear[1].weight, 0)

	# def _set_bn_param(self, bn_momentum, bn_eps):
	# 	for m in self.modules():
	# 		if isinstance(m, nn.BatchNorm2d):
	# 			m.momentum = bn_momentum
	# 			m.eps = bn_eps
