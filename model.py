import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualDenseBlock(nn.Module):
    def __init__(self, in_channels, growth_channels=32):
        super(ResidualDenseBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, growth_channels, 3, 1, 1)
        self.conv2 = nn.Conv2d(in_channels + growth_channels, growth_channels, 3, 1, 1)
        self.conv3 = nn.Conv2d(in_channels + 2 * growth_channels, growth_channels, 3, 1, 1)
        self.conv4 = nn.Conv2d(in_channels + 3 * growth_channels, growth_channels, 3, 1, 1)
        self.conv5 = nn.Conv2d(in_channels + 4 * growth_channels, in_channels, 3, 1, 1)

    def forward(self, x):
        x1 = F.leaky_relu(self.conv1(x), negative_slope=0.2)
        x2 = F.leaky_relu(self.conv2(torch.cat((x, x1), 1)), negative_slope=0.2)
        x3 = F.leaky_relu(self.conv3(torch.cat((x, x1, x2), 1)), negative_slope=0.2)
        x4 = F.leaky_relu(self.conv4(torch.cat((x, x1, x2, x3), 1)), negative_slope=0.2)
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x

class RRDB(nn.Module):
    def __init__(self, in_channels, growth_channels=32):
        super(RRDB, self).__init__()
        self.RDB1 = ResidualDenseBlock(in_channels, growth_channels)
        self.RDB2 = ResidualDenseBlock(in_channels, growth_channels)
        self.RDB3 = ResidualDenseBlock(in_channels, growth_channels)

    def forward(self, x):
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return out * 0.2 + x

class RRDBNet(nn.Module):
    def __init__(self, in_channels, out_channels, num_features, num_blocks, gc=32):
        super(RRDBNet, self).__init__()
        self.conv_first = nn.Conv2d(in_channels, num_features, 3, 1, 1)
        self.RRDB_trunk = nn.Sequential(*[RRDB(num_features, gc) for _ in range(num_blocks)])
        self.trunk_conv = nn.Conv2d(num_features, num_features, 3, 1, 1)
        self.upconv1 = nn.Conv2d(num_features, num_features, 3, 1, 1)
        self.upconv2 = nn.Conv2d(num_features, num_features, 3, 1, 1)
        self.HRconv = nn.Conv2d(num_features, num_features, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_features, out_channels, 3, 1, 1)

    def forward(self, x):
        fea = self.conv_first(x)
        trunk = self.trunk_conv(self.RRDB_trunk(fea))
        fea = fea + trunk
        out = F.leaky_relu(self.upconv1(fea), negative_slope=0.2)
        out = F.leaky_relu(self.upconv2(out), negative_slope=0.2)
        out = F.leaky_relu(self.HRconv(out), negative_slope=0.2)
        out = self.conv_last(out)
        return out
