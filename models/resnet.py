import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

class ResNet(nn.Module):
	def __init__(self, resnet_model, opts):
		super(ResNet, self).__init__()
		if resnet_model == 'ResNet18':
			self.resnet = models.resnet18(weights=None, num_classes=opts.num_classes)
		elif resnet_model == 'ResNet34':
			self.resnet = models.resnet34(weights=None, num_classes=opts.num_classes)
		elif resnet_model == 'ResNet50':
			self.resnet = models.resnet50(weights=None, num_classes=opts.num_classes)

		self.resnet.conv1 = nn.Conv2d(opts.num_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
		self.resnet.fc.register_forward_hook(lambda m, inp, out: F.dropout(out, p=opts.dropout, training=m.training))
		if opts.class_weights:
			weights = 1096/(opts.class_count*3)
			self.criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights))
		else:
			self.criterion = nn.CrossEntropyLoss()


	def forward(self, x):
		return self.resnet(x)

	def loss(self, pred, gt):
		return self.criterion(pred, gt)