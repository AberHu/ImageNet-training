import os
import cv2
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
import nvidia.dali.pipeline as pipeline
import nvidia.dali.ops as ops
import nvidia.dali.types as types
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# If get UserWarning: Corrupt EXIF data, use cv2_loader or ignore warnings
def pil_loader(path):
	img = Image.open(path).convert('RGB')
	return img

def cv2_loader(path):
	img = cv2.imread(path, cv2.IMREAD_COLOR)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	img = Image.fromarray(img)
	return img

default_loader = pil_loader


def default_list_reader(list_path):
	img_list = []
	with open(list_path, 'r') as f:
		for line in f.readlines():
			img_path, label = line.strip().split(' ')
			img_list.append((img_path, int(label)))

	return img_list


class ImageList(data.Dataset):
	def __init__(self, root, list_path, transform=None, list_reader=default_list_reader, loader=default_loader):
		self.root       = root
		self.img_list   = list_reader(list_path)
		self.transform  = transform
		self.loader     = loader

	def __getitem__(self, index):
		img_path, target = self.img_list[index]
		img = self.loader(os.path.join(self.root, img_path))

		if self.transform:
			img = self.transform(img)

		return img, target

	def __len__(self):
		return len(self.img_list)


def get_train_transform(coji=False):
	transform_list = [
			transforms.RandomResizedCrop(224),
			transforms.RandomHorizontalFlip(0.5),
		]
	if coji:
		transform_list += [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),]
	transform_list += [
			transforms.ToTensor(),
			transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
		]
	train_transform = transforms.Compose(transform_list)

	return train_transform


def get_val_transform():
	transform_list = [
		transforms.Resize(256),
		transforms.CenterCrop(224),
		transforms.ToTensor(),
		transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
	]
	val_transform = transforms.Compose(transform_list)

	return val_transform


class HybridTrainPipe(pipeline.Pipeline):
	def __init__(self, batch_size, num_threads, device_id, root, list_path,
				 crop, shard_id, num_shards, coji=False, dali_cpu=False):
		super(HybridTrainPipe, self).__init__(batch_size,
											  num_threads,
											  device_id,
											  seed=12 + device_id)
		self.read = ops.FileReader(file_root=root,
								   file_list=list_path,
								   shard_id=shard_id,
								   num_shards=num_shards,
								   random_shuffle=True,
								   initial_fill=1024)
		# Let user decide which pipeline works
		dali_device    = 'cpu' if dali_cpu else 'gpu'
		decoder_device = 'cpu' if dali_cpu else 'mixed'
		# This padding sets the size of the internal nvJPEG buffers to be able to handle all images
		# from full-sized ImageNet without additional reallocations
		device_memory_padding = 211025920 if decoder_device == 'mixed' else 0
		host_memory_padding   = 140544512 if decoder_device == 'mixed' else 0
		self.decode = ops.ImageDecoderRandomCrop(device=decoder_device, output_type=types.RGB,
												 device_memory_padding=device_memory_padding,
												 host_memory_padding=host_memory_padding,
												 random_aspect_ratio=[0.75, 1.33333333],
												 random_area=[0.08, 1.0],
												 num_attempts=100)
		self.resize = ops.Resize(device=dali_device,
								 resize_x=crop,
								 resize_y=crop,
								 interp_type=types.INTERP_TRIANGULAR)
		self.cmnp = ops.CropMirrorNormalize(device=dali_device,
											output_dtype=types.FLOAT,
											output_layout=types.NCHW,
											crop=(crop, crop),
											image_type=types.RGB,
											mean=[x*255 for x in IMAGENET_MEAN],
											std=[x*255 for x in IMAGENET_STD])
		self.coin = ops.CoinFlip(probability=0.5)

		self.coji = coji
		if self.coji:
			self.twist = ops.ColorTwist(device=dali_device)
			self.brightness_rng = ops.Uniform(range=[1.0-0.4, 1.0+0.4])
			self.contrast_rng = ops.Uniform(range=[1.0-0.4, 1.0+0.4])
			self.saturation_rng = ops.Uniform(range=[1.0-0.4, 1.0+0.4])

	def define_graph(self):
		rng = self.coin()
		imgs, targets = self.read(name="Reader")
		imgs = self.decode(imgs)
		imgs = self.resize(imgs)
		if self.coji:
			brightness = self.brightness_rng()
			contrast = self.contrast_rng()
			saturation = self.saturation_rng()
			imgs = self.twist(imgs, brightness=brightness, contrast=contrast, saturation=saturation)
		imgs = self.cmnp(imgs, mirror=rng)
		return [imgs, targets]


class HybridValPipe(pipeline.Pipeline):
	def __init__(self, batch_size, num_threads, device_id, root, list_path,
				 size, crop, shard_id, num_shards, dali_cpu=False):
		super(HybridValPipe, self).__init__(batch_size,
											  num_threads,
											  device_id,
											  seed=12 + device_id)
		self.read = ops.FileReader(file_root=root,
								   file_list=list_path,
								   shard_id=shard_id,
								   num_shards=num_shards,
								   random_shuffle=False)
		# Let user decide which pipeline works
		dali_device    = 'cpu' if dali_cpu else 'gpu'
		decoder_device = 'cpu' if dali_cpu else 'mixed'
		self.decode = ops.ImageDecoder(device=decoder_device, output_type=types.RGB)
		self.resize = ops.Resize(device=dali_device,
								 resize_shorter=size,
								 interp_type=types.INTERP_TRIANGULAR)
		self.cmnp = ops.CropMirrorNormalize(device=dali_device,
											output_dtype=types.FLOAT,
											output_layout=types.NCHW,
											crop=(crop, crop),
											image_type=types.RGB,
											mean=[x*255 for x in IMAGENET_MEAN],
											std=[x*255 for x in IMAGENET_STD])

	def define_graph(self):
		imgs, targets = self.read(name="Reader")
		imgs = self.decode(imgs)
		imgs = self.resize(imgs)
		imgs = self.cmnp(imgs)
		return [imgs, targets]
