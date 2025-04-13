import torch.nn as nn
from models.alex_net import AlexNet
from models.resnet import ResNet

class ClassifierModel(nn.Module):
    def __init__(self, classfier_type, opts):
        super().__init__()
        if classfier_type == 'AlexNet':
            self.classifier = AlexNet(opts)
        elif 'ResNet' in classfier_type:
            self.classifier = ResNet(classfier_type, opts)

    def forward(self, x):
        return self.classifier.forward(x)

    def loss(self, pred, gt):
        return self.classifier.loss(pred,gt)
