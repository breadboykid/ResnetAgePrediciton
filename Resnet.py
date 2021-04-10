import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ResidualBlock(nn.Module):

    def __init__(self, channels1, channels2, res_stride=1):
        super(ResidualBlock, self).__init__()
        self.inplanes = channels1

        self.bn1 = nn.BatchNorm2d(channels1)
        self.conv1 = nn.Conv2d(channels1, channels2, kernel_size=3, stride=res_stride, padding=1, bias=False)

        self.bn2 = nn.BatchNorm2d(channels2)
        self.conv2 = nn.Conv2d(channels2, channels2, kernel_size=3, stride=1, padding=1, bias=False)

        if res_stride != 1 or channels2 != channels1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(channels1, channels2, kernel_size=1, stride=res_stride, bias=False),
                nn.BatchNorm2d(channels2)
            )
        else:
            self.shortcut = nn.Sequential()

    def forward(self, x):

        # forward pass: BatchNorm2d > ReLU > Conv2d > BatchNorm2d > ReLU > Conv2d > ADD
        out = self.bn1(x)
        out = F.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = F.relu(out)
        out = self.conv2(out)

        out += self.shortcut(x)

        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_strides, num_features, in_channels, num_classes=1):
        super(ResNet, self).__init__()
        self.in_planes = num_features[0]  #  in_planes stores the number channels output from first convolution

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=num_features[0], kernel_size=3,
                               stride=num_strides[0], padding=1, bias=False)

        self.layer1 = self._make_layer(block, num_features[1], num_blocks, stride=num_strides[1])
        self.layer2 = self._make_layer(block, num_features[2], num_blocks, stride=num_strides[2])
        self.layer3 = self._make_layer(block, num_features[3], num_blocks, stride=num_strides[3])

        self.linear = nn.Linear(num_features[3], 1)

    def _make_layer(self, block, planes, num_blocks, stride):
        layers = []
        # create initial layer with option of downsampling and increase channels
        layers.append(block(self.in_planes, planes, stride))

        #  then create num_blocks more for each group
        for i in np.arange(num_blocks):
            layers.append(block(planes, planes))

        # update class attribute in_planes which is keeping track of input channels
        self.in_planes = planes

        return nn.Sequential(*layers)  #  return sequential object comining layers

    def forward(self, x):
        # initial convolution
        out = F.relu(self.conv1(x))

        # residual blocks
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)

        # average pool (flattens spatial dimensions)
        out = F.avg_pool2d(out, (15, 20))
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        return out
