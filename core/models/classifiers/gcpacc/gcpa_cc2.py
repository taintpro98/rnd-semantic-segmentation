import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from core.models.classifiers.gcpacc.contextagg import (
    # SpatialCGNL,
    LocalAttenModule,
    CrissCrossAttention,
    # SmallLocalAttenModule,
)
from torch.nn import BatchNorm2d, BatchNorm1d
from core.models.classifiers.gcpacc.gcpa_gald import FAM
from core.models.classifiers.gcpacc.encoders import hardnet

class GCPAEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.hardnet = hardnet(arch=68)

    def forward(self, x):
        hardnetout = self.hardnet(x)
        return hardnetout

class GCPADecoder(nn.Module):
    def __init__(self, num_classes=19):
        super().__init__()
        inplanes = 1024
        interplanes = 256

        self.fam45 = FAM(640, interplanes, interplanes, interplanes)
        self.fam34 = FAM(320, interplanes, interplanes, interplanes)
        self.fam23 = FAM(128, interplanes, interplanes, interplanes)

        self.linear5 = nn.Conv2d(interplanes, num_classes, kernel_size=3, stride=1, padding=1)
        self.linear4 = nn.Conv2d(interplanes, num_classes, kernel_size=3, stride=1, padding=1)
        self.linear3 = nn.Conv2d(interplanes, num_classes, kernel_size=3, stride=1, padding=1)
        self.linear2 = nn.Conv2d(interplanes, num_classes, kernel_size=3, stride=1, padding=1)

        self.conva = nn.Sequential(
            nn.Conv2d(inplanes, interplanes, 3, padding=1, bias=False),
            BatchNorm2d(interplanes),
            nn.ReLU(interplanes),
        )

        self.long_relation = CrissCrossAttention(interplanes)
        self.local_attention_4 = LocalAttenModule(interplanes)
        self.local_attention_3 = LocalAttenModule(interplanes)
        self.local_attention_2 = LocalAttenModule(interplanes)

    def forward(self, x, hardnetout):
        out2 = hardnetout[0]  # [24, 128, 88, 88]
        out3 = hardnetout[1]  # [24, 320, 44, 44]
        out4 = hardnetout[2]  # [24, 640, 22, 22]
        out5_ = hardnetout[3]  # [24, 1024, 11, 11]

        out5_ = self.conva(out5_)  # bs, 256, 11, 11

        out5_c = self.long_relation(out5_)  # bs, 256, 11, 11
        out5_c = self.long_relation(out5_c)  # bs, 256, 11, 11

        # GCF
        out4_c = self.local_attention_4(out5_c)  # bs, 256, 11, 11

        out3_c = self.local_attention_3(out5_c)  # bs, 256, 11, 11

        out2_c = self.local_attention_2(out5_c)  # bs, 256, 11, 11

        # HA
        # out5 = self.conva(out5_)  # bs, 256, 11, 11

        out5 = out5_  # bs, 256, 11, 11

        # out
        out4 = self.fam45(out4, out5, out4_c)
        out3 = self.fam34(out3, out4, out3_c)
        out2 = self.fam23(out2, out3, out2_c)
        # we use bilinear interpolation instead of transpose convolution
        out5 = F.interpolate(self.linear5(out5), size=x.size()[2:], mode="bilinear")
        out4 = F.interpolate(self.linear4(out4), size=x.size()[2:], mode="bilinear")
        out3 = F.interpolate(self.linear3(out3), size=x.size()[2:], mode="bilinear")
        out2 = F.interpolate(self.linear2(out2), size=x.size()[2:], mode="bilinear")
        return out5, out4, out3, out2

# class GCPACC2Net(nn.Module):
#     def __init__(self):
#         super(GCPACC2Net, self).__init__()

#         self.hardnet = hardnet(arch=68)

#         inplanes = 1024
#         interplanes = 256

#         self.fam45 = FAM(640, interplanes, interplanes, interplanes)
#         self.fam34 = FAM(320, interplanes, interplanes, interplanes)
#         self.fam23 = FAM(128, interplanes, interplanes, interplanes)

#         self.linear5 = nn.Conv2d(interplanes, 1, kernel_size=3, stride=1, padding=1)
#         self.linear4 = nn.Conv2d(interplanes, 1, kernel_size=3, stride=1, padding=1)
#         self.linear3 = nn.Conv2d(interplanes, 1, kernel_size=3, stride=1, padding=1)
#         self.linear2 = nn.Conv2d(interplanes, 1, kernel_size=3, stride=1, padding=1)

#         self.conva = nn.Sequential(
#             nn.Conv2d(inplanes, interplanes, 3, padding=1, bias=False),
#             BatchNorm2d(interplanes),
#             nn.ReLU(interplanes),
#         )

#         self.long_relation = CrissCrossAttention(interplanes)
#         self.local_attention_4 = LocalAttenModule(interplanes)
#         self.local_attention_3 = LocalAttenModule(interplanes)
#         self.local_attention_2 = LocalAttenModule(interplanes)

#     def forward(self, x):
#         hardnetout = self.hardnet(x)
#         # out1 = self.resnet.maxpool(out1)  # bs, 64, 88, 88

#         out2 = hardnetout[0]  # [24, 128, 88, 88]
#         out3 = hardnetout[1]  # [24, 320, 44, 44]
#         out4 = hardnetout[2]  # [24, 640, 22, 22]
#         out5_ = hardnetout[3]  # [24, 1024, 11, 11]

#         out5_ = self.conva(out5_)  # bs, 256, 11, 11

#         out5_c = self.long_relation(out5_)  # bs, 256, 11, 11
#         out5_c = self.long_relation(out5_c)  # bs, 256, 11, 11

#         # GCF
#         out4_c = self.local_attention_4(out5_c)  # bs, 256, 11, 11

#         out3_c = self.local_attention_3(out5_c)  # bs, 256, 11, 11

#         out2_c = self.local_attention_2(out5_c)  # bs, 256, 11, 11

#         # HA
#         # out5 = self.conva(out5_)  # bs, 256, 11, 11

#         out5 = out5_  # bs, 256, 11, 11

#         # out
#         out4 = self.fam45(out4, out5, out4_c)
#         out3 = self.fam34(out3, out4, out3_c)
#         out2 = self.fam23(out2, out3, out2_c)
#         # we use bilinear interpolation instead of transpose convolution
#         out5 = F.interpolate(self.linear5(out5), size=x.size()[2:], mode="bilinear")
#         out4 = F.interpolate(self.linear4(out4), size=x.size()[2:], mode="bilinear")
#         out3 = F.interpolate(self.linear3(out3), size=x.size()[2:], mode="bilinear")
#         out2 = F.interpolate(self.linear2(out2), size=x.size()[2:], mode="bilinear")
#         return out5, out4, out3, out2
