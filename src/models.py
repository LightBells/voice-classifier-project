import torch
import torch.nn as nn
import timm

class Net(nn.Module):
    def __init__(self, out_classes):
        super(Net, self).__init__()
        self.model = timm.create_model('resnet18')
        output_channels = self.model.conv1.out_channels
        self.model.conv1 = nn.Conv2d(1, output_channels,kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, out_classes)

    def forward(self, x):
        x = self.model(x)
        return x
