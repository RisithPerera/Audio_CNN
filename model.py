import torch.nn as nn

class AudioCNN(nn.Module):

    def __init__(self, in_channels, out_channels, stride = 1):
        super().__init()__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, padding = 1, bias = False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv1 = nn.Conv2d(out_channels, out_channels, 3, stride, padding = 1, bias = False)
        self.bn2 = nn.BatchNorm2d(out_channels)