from torch.nn import functional as F
from torch import nn
from torch import sigmoid


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(ResidualBlock, self).__init__()
        self.conv_1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1)
        self.conv_1_bn = nn.BatchNorm2d(out_channels)
        self.conv_2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.conv_2_bn = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.conv_1(x)
        out = self.conv_1_bn(out)
        out = F.relu(out)
        out = self.conv_2(out)
        out = self.conv_2_bn(out)
        out += residual
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.conv = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2)
        self.conv_bn = nn.BatchNorm2d(num_features=64)
        self.conv_maxpool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.resblock_1 = ResidualBlock(64, 64, 1)
        self.resblock_2 = ResidualBlock(64, 128, 2)
        self.resblock_3 = ResidualBlock(128, 256, 2)
        self.resblock_4 = ResidualBlock(256, 512, 2)
        self.GlobabalAvgPool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.fcl = nn.Linear(512, 2)

    def forward(self, x):
        x = self.conv(x)
        x = self.conv_bn(x)
        x = self.conv_maxpool(x)
        x = F.relu(x)
        x = self.resblock_1(x)
        x = self.resblock_2(x)
        x = self.resblock_3(x)
        x = self.resblock_4(x)
        x = self.GlobabalAvgPool(x)
        x = self.flatten(x)
        x = self.fcl(x)
        x = sigmoid(x)
        return x
        