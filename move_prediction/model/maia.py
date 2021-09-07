import torch
import torch.nn as nn
import torch.nn.functional as F


class SE_Block(nn.Module):
	"credits: https://github.com/moskomule/senet.pytorch/blob/master/senet/se_module.py#L4"
	def __init__(self, c, r=16):
		super().__init__()
		self.squeeze = nn.AdaptiveAvgPool2d(1)
		self.excitation = nn.Sequential(
			nn.Linear(c, c // r, bias=False),
			nn.ReLU(inplace=True),
			nn.Linear(c // r, c, bias=False),
			nn.Sigmoid()
		)

	def forward(self, x):
		bs, c, _, _ = x.shape
		y = self.squeeze(x).view(bs, c)
		y = self.excitation(y).view(bs, c, 1, 1)
		return x * y.expand_as(x)


class Maia(nn.Module):
	def __init__(self):
		super(Maia, self).__init__()

		def block(inp, out_1, out_2, ks_1, ks_2, stride, padding, se_r):
			return nn.Sequential(
				nn.Conv2d(in_channels=inp, out_channels=out_1, kernel_size=ks_1, padding=padding, stride=stride),
				nn.BatchNorm2d(out_1),
				nn.ReLU(),
				nn.Conv2d(in_channels=out_1, out_channels=out_2, kernel_size=ks_2, padding=padding, stride=stride),
				SE_Block(c=out_2, r=se_r),
				nn.ReLU(),
			)

		self.blocks = nn.Sequential(
			block(64, 64, 64, 3, 3, 1, 1, 8),
			block(64, 64, 64, 3, 3, 1, 1, 8),
			block(64, 64, 64, 3, 3, 1, 1, 8),
			block(64, 64, 64, 3, 3, 1, 1, 8),
			block(64, 64, 64, 3, 3, 1, 1, 8),
			block(64, 64, 64, 3, 3, 1, 1, 8)
		)


		self.cov1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, padding=1, stride=1)
		self.bn1 = nn.BatchNorm2d(64)
		self.relu = nn.ReLU()
		self.cov2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=1)
		self.cov4 = nn.Conv2d(in_channels=64, out_channels=80, kernel_size=3, padding=1, stride=1)
		self.cov3 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1, stride=1)
		self.fc1 = nn.Linear(1, 128)  # width out * height out * channels out
		self.fc2 = nn.Linear(14*14*32, 128)  # width out * height out * channels out

	def forward(self, x):
		x = self.features(x)
		x = F.max_pool2d(x, kernel_size=2)
		x = x.view(x.size(0), -1)
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.fc3(x)
		return x
