import os
import sys
import time
import torch
import torch.backends.cudnn as cudnn


def compute_speed(model, input_size, device='cuda:0', iteration=1000):
	assert isinstance(input_size, (list, tuple))
	assert len(input_size) == 4
	os.environ['OMP_NUM_THREADS'] = '1'
	os.environ['MKL_NUM_THREADS'] = '1'

	device = torch.device(device)
	if 'cuda' in str(device):
		cudnn.enabled = True
		cudnn.benchmark = True
		torch.cuda.set_device(device)

	model = model.to(device)
	model.eval()

	x = torch.randn(*input_size, device=device)
	x.to(device)

	# warmup for 100 iterations
	for _ in range(100):
		model(x)

	print('=============Speed Testing=============')
	print('Device: {}'.format(str(device)))
	if 'cuda' in str(device):
		torch.cuda.synchronize() # wait for cuda to finish (cuda is asynchronous!)
		torch.cuda.synchronize()
	t_start = time.time()
	for _ in range(iteration):
		model(x)
	if 'cuda' in str(device):
		torch.cuda.synchronize() # wait for cuda to finish (cuda is asynchronous!)
		torch.cuda.synchronize()
	elapsed_time = time.time() - t_start
	print('Elapsed time: [%.2fs / %diter]' % (elapsed_time, iteration))
	print('Speed Time: %.2fms/iter  FPS: %.2f' % (
		elapsed_time / iteration * 1000, iteration * input_size[0] / elapsed_time))


