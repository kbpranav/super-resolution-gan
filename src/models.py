import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg19
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        self.conv_block = nn.Sequential(
        nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, stride=1),
        nn.BatchNorm2d(in_channels, 0.8),
        nn.PReLU(),
        nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, stride=1),
        nn.BatchNorm2d(in_channels, 0.8),
        )

    def forward(self, x):
        return x + self.conv_block(x)


class Generator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, n_blocks = 16):
        super(Generator, self).__init__()

        # First layer
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, 64, kernel_size=9, padding=4), nn.PReLU())

        # Residual blocks
        res_blocks = []
        for _ in range(n_blocks):
            res_blocks.append(ResidualBlock(64))
        self.res_blocks = nn.Sequential(*res_blocks)

        # Second conv layer
        self.conv2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(64, 0.8))

        # Upsampling layers

        upsample = []

        for up in range(2):
            upsample += [
                nn.Conv2d(64, 256, 3, 1, 1),
                nn.BatchNorm2d(256),
                nn.PixelShuffle(2),
                nn.PReLU(),
            ]
        
        self.upsample = nn.Sequential(*upsample)

        # Final layer
        self.conv3 = nn.Sequential(nn.Conv2d(64, out_channels, kernel_size=9, padding=4), nn.Tanh())
    
    def forward(self, x):
        out1 = self.conv1(x)
        out = self.res_blocks(out1)
        out2 = self.conv2(out)
        out = torch.add(out1, out2)
        out = self.upsample(out)
        out = self.conv3(out)
        return out
    
class ExtractFeatures(nn.Module):
    def __init__(self):
        super(ExtractFeatures, self).__init__()
        vgg = vgg19(pretrained=True)
        # self.vgg = nn.Sequential(*list(vgg.features.childern())[:18])
        self.feature_extractor = nn.Sequential(*list(vgg.features.children())[:18])
    def forward(self, x):
        # return self.vgg(x).
        return self.feature_extractor(x)


class Discriminator(nn.Module):
    def __init__(self, input_shape):
        super(Discriminator, self).__init__()
        self.input_shape = input_shape
        in_channels, in_height, in_width = input_shape

        patch_h, patch_w = int(in_height / 2**4), int(in_width / 2**4)
        self.output_shape = (1, patch_h, patch_w)

        def discriminator_block(in_filters, out_filters, first_block=False):
            layers = []
            layers.append(nn.Conv2d(in_filters, out_filters, kernel_size=3, stride=1, padding=1))
            if not first_block:
                layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            layers.append(nn.Conv2d(out_filters, out_filters, kernel_size=3, stride=2, padding=1))
            layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        layers = []
        in_filters = in_channels
        for i, out_filters in enumerate([64, 128, 256, 512]):
            layers.extend(discriminator_block(in_filters, out_filters, first_block=(i==0)))
            in_filters = out_filters

        layers.append(nn.Conv2d(out_filters, 1, kernel_size=3, stride=1, padding=1))
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)