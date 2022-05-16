import torch
import torch.nn as nn
import torch.nn.functional as F

class Block(nn.Module):
    def __init__(self, outer_dim, inner_dim, repr_dim=None):
        super(Block, self).__init__()

        if repr_dim is None:
            repr_dim = inner_dim
        self.conv_inner = nn.Conv2d(in_channels=outer_dim, out_channels=inner_dim, kernel_size=1)
        self.conv_loc = nn.Conv2d(in_channels=inner_dim, out_channels=inner_dim, kernel_size=3, padding='same')
        self.conv_dil = nn.Conv2d(in_channels=inner_dim, out_channels=inner_dim, kernel_size=3, dilation=3, padding='same')
        self.conv_delta = nn.Conv2d(in_channels=inner_dim*2, out_channels=outer_dim, kernel_size=1)
    
    def forward(self, x):
        v = F.relu(self.conv_inner(x))
        a = F.relu(self.conv_loc(v))
        b = F.relu(self.conv_loc(v))
        v = torch.cat([a, b], axis=1) # TODO is this the right axis to cat on?
        x = x + self.conv_delta(v)
        return x

class MeanColorModelV2(nn.Module):
    def __init__(self):
        super(MeanColorModelV2, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, padding='same')
        self.blocks = nn.ModuleList()

        for i in range(5):
          self.blocks.append(Block(64, 48))
        self.blocks.append(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding='same'))

        for i in range(6):
          self.blocks.append(Block(128, 96))
        self.blocks.append(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding='same'))

        for i in range(3):
          self.blocks.append(Block(256, 192))
        self.convcolor = nn.Conv2d(in_channels=256, out_channels=3, kernel_size=1)
        #self.convfactor = nn.Conv2D(in_channels=256, out_channels=256, kernel_size=1)
        #self.converr = nn.Conv2D(in_channels=256, out_channels=3, kernel_size=1)

    def forward(self, x):
        x = self.conv1(x)
        for block in self.blocks:
          x = block(x)
        out = self.convcolor(x)

        #factor = self.convfactor(x)
        #err = self.converr(x)
        return out#, factor, err*err + 0.01