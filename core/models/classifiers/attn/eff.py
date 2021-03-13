import torch
import torch.nn as nn
import torch.nn.functional as F
# import pytorch_lightning as pl

from torchvision.models import ResNet
from loguru import logger
from efficientnet_pytorch import EfficientNet
from efficientnet_pytorch.utils import efficientnet_params
from typing import Callable, Optional

from polypnet.model.attn import AdditiveAttnGate

class Encoder(nn.Module):
    def __init__(self, backbone_name="efficientnet-b0", num_classes=1):
        self.encoder = EfficientNet.from_pretrained(backbone_name)

    def forward(self, inputs):
        endpoints = self.encoder.extract_endpoints(inputs)
        return endpoints

class Decoder(nn.Module):
    def __init__(self,
        backbone_name="efficientnet-b0",
        num_classes=1
    ):
        super().__init__()

        self.num_classes = num_classes

        self.d5, self.d4, self.d3, self.d2, self.d1, self.d0 = self._block_depths(backbone_name)

        self._init_upsamplers()

        self.decode_4 = self._decoder_block(self.d4 * 2, self.d4)
        self.decode_3 = self._decoder_block(self.d3 * 2, self.d3)
        self.decode_2 = self._decoder_block(self.d2 * 2, self.d2)
        self.decode_1 = self._decoder_block(self.d1 * 2, self.d1)
        self.decode_0 = nn.Sequential(
            nn.Conv2d(self.d0, self.d0 // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.d0 // 2),
            nn.LeakyReLU()
        )

        self.out_4 = self._out_block(self.d4)
        self.out_3 = self._out_block(self.d3)
        self.out_2 = self._out_block(self.d2)
        self.out_1 = self._out_block(self.d1)
        self.out_0 = self._out_block(self.d0)

        self._init_attn_blocks()

    def _init_upsamplers(self):
        self.mid_upsampler = nn.ConvTranspose2d(in_channels=self.d5, out_channels=self.d4, kernel_size=4, stride=2, padding=1, bias=False)
        self.ups_4 = self._upsampler_block(in_channels=self.d4, out_channels=self.d3)
        self.ups_3 = self._upsampler_block(in_channels=self.d3, out_channels=self.d2)
        self.ups_2 = self._upsampler_block(in_channels=self.d2, out_channels=self.d1)
        self.ups_1 = self._upsampler_block(in_channels=self.d1, out_channels=self.d0)

    def _init_attn_blocks(self):
        self.attn_mid = AdditiveAttnGate(self.d5, self.d4)
        self.attn_4 = AdditiveAttnGate(self.d4, self.d3)
        self.attn_3 = AdditiveAttnGate(self.d3, self.d2)
        self.attn_2 = AdditiveAttnGate(self.d2, self.d1)

    def _block_depths(self, backbone_name):
        depth_map = {
            "efficientnet-b0": (1280, 112, 40, 24, 16, 8),
            "efficientnet-b1": (1280, 112, 40, 24, 16, 8),
            "efficientnet-b2": (1408, 120, 48, 24, 16, 8),
            "efficientnet-b3": (1536, 136, 48, 32, 24, 12),
            "efficientnet-b4": (1792, 160, 56, 32, 24, 12),
        }
        return depth_map[backbone_name]

    def _out_block(self, in_channels: int):
        return nn.Sequential(
            nn.Conv2d(in_channels, self.num_classes, kernel_size=1),
        )

    def _decoder_block(self, in_channels: int, out_channels: int):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels // 2),
            nn.LeakyReLU(),
            nn.Conv2d(out_channels // 2, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU()
        )

    def _upsampler_block(self, in_channels, out_channels):
        return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False)

    @property
    def output_scales(self):
        return 1., 1/2, 1/4, 1/8, 1/16

    def forward(self, endpoints):
        """Forward method

        :param inputs: Input tensor of size (B x C x W x H)
        :type inputs: list
        :return: List of output for each level
        """

        # Run input through encoder
        encode_4 = endpoints["reduction_4"]  # B x 112 x H16 x W16
        encode_3 = endpoints["reduction_3"]  # B x 40 x H8 x W8
        encode_2 = endpoints["reduction_2"]  # B x 24 x H4 x W4
        encode_1 = endpoints["reduction_1"]  # B x 16 x H2 x W2

        # Upsample middle block
        middle_block = endpoints["reduction_5"]   # B x 1280 x H32 x W32
        attn_middle = self.attn_mid(middle_block, encode_4)  # B x 1280 x H32 x W32
        up_middle = self.mid_upsampler(attn_middle)  # B x 112 x H16 x W16

        # Level 4
        merged_4 = torch.cat((encode_4, up_middle), dim=1)  # B x 224 x H16 x W16
        decode_4 = self.decode_4(merged_4)  # B x 40 x H16 x W16
        attn_4 = self.attn_4(decode_4, encode_3)  # B x 40 x H16 x W16
        out_4 = self.out_4(decode_4)  # B x 2 x H16 x W16
        up_4 = self.ups_4(attn_4)  # B x 40 x H8 x W8

        # Level 3
        merged_3 = torch.cat((encode_3, up_4), dim=1)  # B x 80 x H8 x W8
        decode_3 = self.decode_3(merged_3)  # B x 40 x H8 x W8
        attn_3 = self.attn_3(decode_3, encode_2)  # B x 40 x H8 x W8
        out_3 = self.out_3(decode_3)  # B x 2 x H8 x W8
        up_3 = self.ups_3(attn_3) # B x 24 x H4 x W4

        # Level 2
        merged_2 = torch.cat((encode_2, up_3), dim=1)  # B x 48 x H4 x W4
        decode_2 = self.decode_2(merged_2)  # B x 24 x H4 x W4
        attn_2 = self.attn_2(decode_2, encode_1)  # B x 24 x H4 x W4
        out_2 = self.out_2(decode_2)  # B x 2 x H4 x W4
        up_2 = self.ups_2(attn_2) # B x 16 x H2 x W2

        # Level 1
        merged_1 = torch.cat((encode_1, up_2), dim=1)  # B x 32 x H2 x W2
        decode_1 = self.decode_1(merged_1)  # B x 16 x H2 x W2
        out_1 = self.out_1(decode_1)  # B x 2 x H2 x W2
        up_1 = self.ups_1(decode_1) # B x 8 x H x W

        # Level 0
        out_0 = self.out_0(up_1)  # B x C x H x W

        return out_0, out_1, out_2, out_3, out_4


def test_1():
    x = torch.randn((5, 3, 224, 224))
    net = AttnEfficientNetUnet(backbone_name="efficientnet-b4")
    outputs = net(x)

    for i, o in enumerate(outputs):
        print(f"out-{i}: {o.shape}")


if __name__ == "__main__":
    test_1()
