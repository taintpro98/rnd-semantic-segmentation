import torch
import torch.nn as nn
import torch.nn.functional as F

class AdditiveAttnGate(nn.Module):
    def __init__(self, x_channels: int, g_channels: int):
        super().__init__()

        self.conv_g = nn.Conv2d(
            g_channels, g_channels, kernel_size=1,
            groups=1,
            bias=True
        )
        self.conv_x = nn.Conv2d(
            x_channels, g_channels, kernel_size=1,
            groups=1,
            bias=False
        )
        self.conv_group = nn.Conv2d(
            g_channels, x_channels, kernel_size=1,
            groups=1, bias=True
        )

    def forward(self, x, g):
        """
        Forward function

        :param x: Finer output signal, tensor of shape B x C x H x W
        :param g: Coarser output signal, tensor of shape B x C x 2H x 2W
        :return: Filtered output signal (B x C x H x W)
        """
        up_x = F.interpolate(x, scale_factor=2, mode="bilinear")  # B x C x 2H x 2W

        out_g = self.conv_g(g)  # B x C x 2H x 2W
        out_x = self.conv_x(up_x)  # B x C x 2H x 2W
        alpha = out_g + out_x  # B x C x 2H x 2W

        alpha = nn.ReLU()(alpha)  # B x C x 2H x 2W
        alpha = self.conv_group(alpha)  # B x C x 2H x 2W
        alpha = nn.Sigmoid()(alpha)  # B x C x 2H x 2W

        alpha = F.interpolate(alpha, scale_factor=0.5, mode="bilinear")  # B x C x H x W

        out = alpha * x

        return out


def test_1():
    attn = AdditiveAttnGate(64, 32)
    x = torch.randn((4, 64, 50, 50))
    g = torch.randn((4, 32, 100, 100))
    out = attn(x, g)
    print(out.shape)

if __name__ == "__main__":
    test_1()
