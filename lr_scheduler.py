import torch
from torch.optim.lr_scheduler import LambdaLR
import warnings


class LambdaLRWithMin(LambdaLR):
	def __init__(self, optimizer, lr_lambda, eta_min=0, last_epoch=-1):
		self.eta_min = eta_min
		super(LambdaLRWithMin, self).__init__(optimizer, lr_lambda, last_epoch)

	def get_lr(self):
		if not self._get_lr_called_within_step:
			warnings.warn("To get the last learning rate computed by the scheduler, "
						  "please use `get_last_lr()`.")

		return [base_lr * lmbda(self.last_epoch) + self.eta_min * (1.0 - lmbda(self.last_epoch))
				for lmbda, base_lr in zip(self.lr_lambdas, self.base_lrs)]
