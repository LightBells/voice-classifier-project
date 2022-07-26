import torch
import torch.nn as nn

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        x = x.reshape(x.shape[:2])
        return x

class Unit(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, stride, padding=0):
        super(Unit, self).__init__()
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn   = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class Net(nn.Module):
    def __init__(self, out_classes):
        super(Net, self).__init__()

        conv_nxn = lambda n: nn.Sequential(
            Unit(1, 32, (1, n), (1, 2), (0, (n-1)//2)),
            Unit(32, 32, (n, 1), (2, 1), ((n-1)//2, 0)),
            Unit(32, 64, (1, n), (1, 2), (0, (n-1)//2)),
            Unit(64, 64, (n, 1), (2, 1), ((n-1)//2, 0)),
        )
        
        self.conv_5x5 = conv_nxn(5)
        self.conv_9x9 = conv_nxn(9)
        self.conv_17x17 = conv_nxn(17)
        self.conv_33x33 = conv_nxn(33)

        self.fe_list = [self.conv_5x5, self.conv_9x9, self.conv_17x17, self.conv_33x33]

        self.global_conv = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=(1, 17), stride=(1, 2), padding=(0, 8)),
            nn.Conv2d(512, 512, kernel_size=(17, 1), stride=(2, 1), padding=(8, 0)),
        )
        self.relu = nn.ReLU()

        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = Flatten()
        self.fc  = nn.Linear(512, out_classes)

    def forward(self, x):
        z = None
        for u in self.fe_list:
            tmp = u(x)
            if z is None:
                z = tmp
            else:
                z = torch.cat((z, tmp), dim=1)

        z = self.global_conv(z)
        z = self.relu(z)
        z = self.gap(z)
        z = self.flatten(z)
        y = self.fc(z)
        return y
