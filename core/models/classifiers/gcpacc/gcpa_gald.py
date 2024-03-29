import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from core.models.classifiers.gcpacc.contextagg import GALDHead, GALDBlock
from torch.nn import BatchNorm2d, BatchNorm1d
from core.models.classifiers.gcpacc.encoders import res2net50_v1b_26w_4s

class CA(nn.Module):
    def __init__(self, in_channel_left, in_channel_down):
        super(CA, self).__init__()
        self.conv0 = nn.Conv2d(in_channel_left, 256, kernel_size=1, stride=1, padding=0)
        self.bn0 = nn.BatchNorm2d(256)
        self.conv1 = nn.Conv2d(in_channel_down, 256, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)

    def forward(self, left, down):
        left = F.relu(self.bn0(self.conv0(left)), inplace=True)  # 256
        down = down.mean(dim=(2, 3), keepdim=True)
        down = F.relu(self.conv1(down), inplace=True)
        down = torch.sigmoid(self.conv2(down))
        return left * down


""" Self Refinement Module """


class SRM(nn.Module):
    def __init__(self, in_channel):
        super(SRM, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, 256, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(256)
        self.conv2 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        out1 = F.relu(self.bn1(self.conv1(x)), inplace=True)  # 256
        out2 = self.conv2(out1)
        w, b = out2[:, :256, :, :], out2[:, 256:, :, :]

        return F.relu(w * out1 + b, inplace=True)


""" Feature Interweaved Aggregation Module """


class FAM(nn.Module):
    def __init__(
        self, in_channel_left, in_channel_down, in_channel_right, interplanes=256
    ):
        super(FAM, self).__init__()
        # self.conv0 = nn.Conv2d(in_channel_left, 256, kernel_size=1, stride=1, padding=0)
        self.conv0 = nn.Conv2d(
            in_channel_left, interplanes, kernel_size=3, stride=1, padding=1
        )
        self.bn0 = nn.BatchNorm2d(interplanes)
        self.conv1 = nn.Conv2d(
            in_channel_down, interplanes, kernel_size=3, stride=1, padding=1
        )
        self.bn1 = nn.BatchNorm2d(interplanes)
        self.conv2 = nn.Conv2d(
            in_channel_right, interplanes, kernel_size=3, stride=1, padding=1
        )
        self.bn2 = nn.BatchNorm2d(interplanes)

        self.conv_d1 = nn.Conv2d(
            interplanes, interplanes, kernel_size=3, stride=1, padding=1
        )
        self.conv_d2 = nn.Conv2d(
            interplanes, interplanes, kernel_size=3, stride=1, padding=1
        )
        self.conv_l = nn.Conv2d(
            interplanes, interplanes, kernel_size=3, stride=1, padding=1
        )
        self.conv3 = nn.Conv2d(
            interplanes * 3, interplanes, kernel_size=3, stride=1, padding=1
        )
        self.bn3 = nn.BatchNorm2d(interplanes)

    def forward(self, left, down, right):
        left = F.relu(self.bn0(self.conv0(left)), inplace=True)  # 256 channels
        down = F.relu(self.bn1(self.conv1(down)), inplace=True)  # 256 channels
        right = F.relu(self.bn2(self.conv2(right)), inplace=True)  # 256

        down_1 = self.conv_d1(down)

        w1 = self.conv_l(left)
        if down.size()[2:] != left.size()[2:]:
            down_ = F.interpolate(down, size=left.size()[2:], mode="bilinear")
            z1 = F.relu(w1 * down_, inplace=True)
        else:
            z1 = F.relu(w1 * down, inplace=True)  # down is mask

        if down_1.size()[2:] != left.size()[2:]:
            down_1 = F.interpolate(down_1, size=left.size()[2:], mode="bilinear")

        z2 = F.relu(down_1 * left, inplace=True)  # left is mask

        # z3
        down_2 = self.conv_d2(right)
        if down_2.size()[2:] != left.size()[2:]:
            down_2 = F.interpolate(down_2, size=left.size()[2:], mode="bilinear")
        z3 = F.relu(down_2 * left, inplace=True)  # down_2 is mask

        out = torch.cat((z1, z2, z3), dim=1)
        return F.relu(self.bn3(self.conv3(out)), inplace=True)


class FAMAG(nn.Module):
    def __init__(
        self, in_channel_left, in_channel_down, in_channel_right, interplanes=256
    ):
        super(FAMAG, self).__init__()
        # self.conv0 = nn.Conv2d(in_channel_left, 256, kernel_size=1, stride=1, padding=0)

        self.conv_l0 = nn.Conv2d(
            in_channel_left, interplanes, kernel_size=1, stride=1, padding=1
        )
        self.bnl0 = nn.BatchNorm2d(interplanes)
        self.conv_d0 = nn.Conv2d(
            in_channel_down, interplanes, kernel_size=1, stride=1, padding=1
        )
        self.bnd0 = nn.BatchNorm2d(interplanes)

        self.conv_l1 = nn.Conv2d(
            in_channel_left, interplanes, kernel_size=1, stride=1, padding=1
        )
        self.bnl1 = nn.BatchNorm2d(interplanes)
        self.conv_d1 = nn.Conv2d(
            in_channel_down, interplanes, kernel_size=1, stride=1, padding=1
        )
        self.bnd1 = nn.BatchNorm2d(interplanes)

        self.conv_l2 = nn.Conv2d(
            in_channel_left, interplanes, kernel_size=1, stride=1, padding=1
        )
        self.bnl2 = nn.BatchNorm2d(interplanes)

        self.conv_r2 = nn.Conv2d(
            in_channel_right, interplanes, kernel_size=1, stride=1, padding=1
        )
        self.bnr2 = nn.BatchNorm2d(interplanes)

        self.psi_1 = nn.Sequential(
            nn.Conv2d(interplanes, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.psi_2 = nn.Sequential(
            nn.Conv2d(interplanes, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.psi_3 = nn.Sequential(
            nn.Conv2d(interplanes, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )

        self.conv_out = nn.Conv2d(
            interplanes * 3, interplanes, kernel_size=3, stride=1, padding=1
        )
        self.bn_out = nn.BatchNorm2d(interplanes)

    def forward(self, left, down, right):

        # BRANCH 1: LOW GUIDE HIGH
        left1 = self.bnl0(self.conv_l0(left))  # 256 channels
        down1 = self.bnd0(self.conv_d0(down))  # 256 channels

        if down1.size()[2:] != left1.size()[2:]:
            down1 = F.interpolate(down1, size=left1.size()[2:], mode="bilinear")
        psi_1 = F.relu(left1 + down1)
        psi_1 = self.psi_1(psi_1)
        zdl = down1 * psi_1

        # BRANCH 2: HIGH GUIDE LOW
        left2 = self.bnl1(self.conv_l1(left))  # 256 channels
        down2 = self.bnd1(self.conv_d1(down))  # 256 channels
        if down2.size()[2:] != left2.size()[2:]:
            down2 = F.interpolate(down2, size=left2.size()[2:], mode="bilinear")
        psi_2 = F.relu(left2 + down2)
        psi_2 = self.psi_2(psi_2)
        zld = left2 * psi_2
        # z2 = F.relu(down_1 * left2, inplace=True)  # left is mask

        # BRANCH 3: CONTEXT GUIDE LOW

        left3 = self.bnl2(self.conv_l2(left))  # 256 channels
        right3 = self.bnr2(self.conv_r2(right))  # 256

        if right3.size()[2:] != left3.size()[2:]:
            right3 = F.interpolate(right3, size=left3.size()[2:], mode="bilinear")

        psi_3 = F.relu(left3 + right3)
        psi_3 = self.psi_3(psi_3)
        zlr = left3 * psi_3

        out = torch.cat((zdl, zld, zlr), dim=1)
        return F.relu(self.bn_out(self.conv_out(out)), inplace=True)


class FAMAGv3(nn.Module):
    def __init__(
        self, in_channel_left, in_channel_down, in_channel_right, interplanes=256
    ):
        super(FAMAGv3, self).__init__()
        # self.conv0 = nn.Conv2d(in_channel_left, 256, kernel_size=1, stride=1, padding=0)

        self.conv_l0 = nn.Conv2d(
            in_channel_left, interplanes, kernel_size=1, stride=1, padding=1
        )
        self.bnl0 = nn.BatchNorm2d(interplanes)
        self.conv_d0 = nn.Conv2d(
            in_channel_down, interplanes, kernel_size=1, stride=1, padding=1
        )
        self.bnd0 = nn.BatchNorm2d(interplanes)

        self.conv_l1 = nn.Conv2d(
            in_channel_left, interplanes, kernel_size=1, stride=1, padding=1
        )
        self.bnl1 = nn.BatchNorm2d(interplanes)
        self.conv_d1 = nn.Conv2d(
            in_channel_down, interplanes, kernel_size=1, stride=1, padding=1
        )
        self.bnd1 = nn.BatchNorm2d(interplanes)

        self.conv_l2 = nn.Conv2d(
            in_channel_left, interplanes, kernel_size=1, stride=1, padding=1
        )
        self.bnl2 = nn.BatchNorm2d(interplanes)

        self.conv_r2 = nn.Conv2d(
            in_channel_right, interplanes, kernel_size=1, stride=1, padding=1
        )
        self.bnr2 = nn.BatchNorm2d(interplanes)

        self.psi_1 = nn.Sequential(
            nn.Conv2d(interplanes, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.psi_2 = nn.Sequential(
            nn.Conv2d(interplanes, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.psi_3 = nn.Sequential(
            nn.Conv2d(interplanes, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )

        self.conv_out = nn.Conv2d(
            interplanes * 3, interplanes, kernel_size=3, stride=1, padding=1
        )
        self.bn_out = nn.BatchNorm2d(interplanes)

    def forward(self, left, down, right):

        # BRANCH 1: LOW GUIDE HIGH
        left1_ = self.bnl0(self.conv_l0(left))  # 256 channels
        down1 = self.bnd0(self.conv_d0(down))  # 256 channels

        if down1.size()[2:] != left1_.size()[2:]:
            left1 = F.interpolate(left1_, size=down1.size()[2:], mode="bilinear")
        psi_1 = F.relu(left1 + down1)
        psi_1 = self.psi_1(psi_1)
        zdl = down1 * psi_1
        zdl = F.interpolate(zdl, size=left1_.size()[2:], mode="bilinear")

        # BRANCH 2: HIGH GUIDE LOW
        left2 = self.bnl1(self.conv_l1(left))  # 256 channels
        down2 = self.bnd1(self.conv_d1(down))  # 256 channels
        if down2.size()[2:] != left2.size()[2:]:
            down2 = F.interpolate(down2, size=left2.size()[2:], mode="bilinear")
        psi_2 = F.relu(left2 + down2)
        psi_2 = self.psi_2(psi_2)
        zld = left2 * psi_2
        # z2 = F.relu(down_1 * left2, inplace=True)  # left is mask

        # BRANCH 3: CONTEXT GUIDE LOW

        left3 = self.bnl2(self.conv_l2(left))  # 256 channels
        right3 = self.bnr2(self.conv_r2(right))  # 256

        if right3.size()[2:] != left3.size()[2:]:
            right3 = F.interpolate(right3, size=left3.size()[2:], mode="bilinear")

        psi_3 = F.relu(left3 + right3)
        psi_3 = self.psi_3(psi_3)
        zlr = left3 * psi_3

        out = torch.cat((zdl, zld, zlr), dim=1)
        return F.relu(self.bn_out(self.conv_out(out)), inplace=True)


class FAMAGv2(nn.Module):
    def __init__(
        self, in_channel_left, in_channel_down, in_channel_right, interplanes=256
    ):
        super(FAMAGv2, self).__init__()
        # self.conv0 = nn.Conv2d(in_channel_left, 256, kernel_size=1, stride=1, padding=0)

        self.conv_l0 = nn.Conv2d(
            in_channel_left, interplanes, kernel_size=3, stride=1, padding=1
        )
        self.bnl0 = nn.BatchNorm2d(interplanes)
        self.conv_d0 = nn.Conv2d(
            in_channel_down, interplanes, kernel_size=3, stride=1, padding=1
        )
        self.bnd0 = nn.BatchNorm2d(interplanes)

        self.conv_l1 = nn.Conv2d(
            in_channel_left, interplanes, kernel_size=3, stride=1, padding=1
        )
        self.bnl1 = nn.BatchNorm2d(interplanes)
        self.conv_d1 = nn.Conv2d(
            in_channel_down, interplanes, kernel_size=3, stride=1, padding=1
        )
        self.bnd1 = nn.BatchNorm2d(interplanes)

        self.conv_l2 = nn.Conv2d(
            in_channel_left, interplanes, kernel_size=3, stride=1, padding=1
        )
        self.bnl2 = nn.BatchNorm2d(interplanes)

        self.conv_r2 = nn.Conv2d(
            in_channel_right, interplanes, kernel_size=3, stride=1, padding=1
        )
        self.bnr2 = nn.BatchNorm2d(interplanes)

        self.psi_1 = nn.Sequential(
            nn.Conv2d(interplanes, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.psi_2 = nn.Sequential(
            nn.Conv2d(interplanes, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.psi_3 = nn.Sequential(
            nn.Conv2d(interplanes, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )

        self.conv_out = nn.Conv2d(
            interplanes * 3, interplanes, kernel_size=3, stride=1, padding=1
        )
        self.bn_out = nn.BatchNorm2d(interplanes)

    def forward(self, left, down, right):

        # BRANCH 1: LOW GUIDE HIGH
        left1 = self.bnl0(self.conv_l0(left))  # 256 channels
        down1 = self.bnd0(self.conv_d0(down))  # 256 channels

        if down1.size()[2:] != left1.size()[2:]:
            down1 = F.interpolate(down1, size=left1.size()[2:], mode="bilinear")
        psi_1 = F.relu(left1 + down1)
        psi_1 = self.psi_1(psi_1)
        zdl = down1 * psi_1

        # BRANCH 2: HIGH GUIDE LOW
        left2 = self.bnl1(self.conv_l1(left))  # 256 channels
        down2 = self.bnd1(self.conv_d1(down))  # 256 channels
        if down2.size()[2:] != left2.size()[2:]:
            down2 = F.interpolate(down2, size=left2.size()[2:], mode="bilinear")
        psi_2 = F.relu(left2 + down2)
        psi_2 = self.psi_2(psi_2)
        zld = left2 * psi_2
        # z2 = F.relu(down_1 * left2, inplace=True)  # left is mask

        # BRANCH 3: CONTEXT GUIDE LOW

        left3 = self.bnl2(self.conv_l2(left))  # 256 channels
        right3 = self.bnr2(self.conv_r2(right))  # 256

        if right3.size()[2:] != left3.size()[2:]:
            right3 = F.interpolate(right3, size=left3.size()[2:], mode="bilinear")

        psi_3 = F.relu(left3 + right3)
        psi_3 = self.psi_3(psi_3)
        zlr = left3 * psi_3

        out = torch.cat((zdl, zld, zlr), dim=1)
        return F.relu(self.bn_out(self.conv_out(out)), inplace=True)


class FAMPra(nn.Module):
    def __init__(
        self, in_channel_left, in_channel_down, in_channel_right, interplanes=256
    ):
        super(FAMPra, self).__init__()
        # self.conv0 = nn.Conv2d(in_channel_left, 256, kernel_size=1, stride=1, padding=0)
        self.in_channel_left = in_channel_left
        self.in_channel_down = in_channel_down
        self.in_channel_right = in_channel_right
        self.conv0 = nn.Conv2d(
            in_channel_left, interplanes, kernel_size=3, stride=1, padding=1
        )
        self.bn0 = nn.BatchNorm2d(interplanes)
        self.conv1 = nn.Conv2d(
            in_channel_down, interplanes, kernel_size=3, stride=1, padding=1
        )
        self.bn1 = nn.BatchNorm2d(interplanes)
        self.conv2 = nn.Conv2d(
            in_channel_right, interplanes, kernel_size=3, stride=1, padding=1
        )
        self.bn2 = nn.BatchNorm2d(interplanes)

        self.conv_d1 = nn.Conv2d(
            interplanes, interplanes, kernel_size=3, stride=1, padding=1
        )
        self.conv_d2 = nn.Conv2d(
            interplanes, interplanes, kernel_size=3, stride=1, padding=1
        )
        self.conv_l = nn.Conv2d(
            interplanes, interplanes, kernel_size=3, stride=1, padding=1
        )
        self.conv3 = nn.Conv2d(
            interplanes * 3, interplanes, kernel_size=3, stride=1, padding=1
        )
        self.bn3 = nn.BatchNorm2d(interplanes)

        self.linear = nn.Conv2d(interplanes, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, left, down, right, crop):
        # print(left.shape, down.shape, right.shape, crop.shape, "iiiii")

        left = F.relu(self.bn0(self.conv0(left)), inplace=True)  # 256 channels
        down = F.relu(self.bn1(self.conv1(down)), inplace=True)  # 256 channels
        right = F.relu(self.bn2(self.conv2(right)), inplace=True)  # 256

        down_1 = self.conv_d1(down)

        w1 = self.conv_l(left)
        if down.size()[2:] != left.size()[2:]:
            down_ = F.interpolate(down, size=left.size()[2:], mode="bilinear")
            z1 = F.relu(w1 * down_, inplace=True)
        else:
            z1 = F.relu(w1 * down, inplace=True)  # down is mask

        if down_1.size()[2:] != left.size()[2:]:
            down_1 = F.interpolate(down_1, size=left.size()[2:], mode="bilinear")

        z2 = F.relu(down_1 * left, inplace=True)  # left is mask

        # z3
        down_2 = self.conv_d2(right)
        if down_2.size()[2:] != left.size()[2:]:
            down_2 = F.interpolate(down_2, size=left.size()[2:], mode="bilinear")
        z3 = F.relu(down_2 * left, inplace=True)  # down_2 is mask

        out = torch.cat((z1, z2, z3), dim=1)

        out = F.relu(self.bn3(self.conv3(out)), inplace=True)  # bs, 256, 22 , 22

        x = -1 * (torch.sigmoid(crop)) + 1  # [bs, 1, 22, 22]
        x = x.expand(-1, left.size()[1], -1, -1)  # [bs, in_channel_left, 22, 22]
        out = x.mul(out)

        ra_feat = self.linear(out)
        return out, ra_feat


class FAMSCWS(nn.Module):
    def __init__(
        self, in_channel_left, in_channel_down, in_channel_right, interplanes=256
    ):
        super(FAMSCWS, self).__init__()
        # self.conv0 = nn.Conv2d(in_channel_left, 256, kernel_size=1, stride=1, padding=0)
        self.conv0 = nn.Conv2d(
            in_channel_left, interplanes, kernel_size=3, stride=1, padding=1
        )
        self.bn0 = nn.BatchNorm2d(interplanes)
        self.conv1 = nn.Conv2d(
            in_channel_down, interplanes, kernel_size=3, stride=1, padding=1
        )
        self.bn1 = nn.BatchNorm2d(interplanes)
        self.conv2 = nn.Conv2d(
            in_channel_right, interplanes, kernel_size=3, stride=1, padding=1
        )
        self.bn2 = nn.BatchNorm2d(interplanes)

        self.conv_d1 = nn.Conv2d(
            interplanes, interplanes, kernel_size=3, stride=1, padding=1
        )
        self.conv_d2 = nn.Conv2d(
            interplanes, interplanes, kernel_size=3, stride=1, padding=1
        )
        self.conv_l = nn.Conv2d(
            interplanes, interplanes, kernel_size=3, stride=1, padding=1
        )
        self.conv3 = nn.Conv2d(
            interplanes, interplanes, kernel_size=3, stride=1, padding=1
        )
        self.bn3 = nn.BatchNorm2d(interplanes)

        self.conv_att1 = nn.Conv2d(interplanes, 1, kernel_size=3, stride=1, padding=1)	
        self.conv_att2 = nn.Conv2d(interplanes, 1, kernel_size=3, stride=1, padding=1)	
        self.conv_att3 = nn.Conv2d(interplanes, 1, kernel_size=3, stride=1, padding=1)	

    def forward(self, left, down, right):
        left = F.relu(self.bn0(self.conv0(left)), inplace=True)  # 256 channels
        down = F.relu(self.bn1(self.conv1(down)), inplace=True)  # 256 channels
        right = F.relu(self.bn2(self.conv2(right)), inplace=True)  # 256

        down_1 = self.conv_d1(down)

        w1 = self.conv_l(left)
        if down.size()[2:] != left.size()[2:]:
            down_ = F.interpolate(down, size=left.size()[2:], mode="bilinear")
            z1 = F.relu(w1 * down_, inplace=True)
        else:
            z1 = F.relu(w1 * down, inplace=True)  # down is mask

        z1_att = F.adaptive_avg_pool2d(self.conv_att1(z1), (1,1))	
        z1 = z1_att * z1	

        if down_1.size()[2:] != left.size()[2:]:
            down_1 = F.interpolate(down_1, size=left.size()[2:], mode="bilinear")

        z2 = F.relu(down_1 * left, inplace=True)  # left is mask
        z2_att = F.adaptive_avg_pool2d(self.conv_att2(z2), (1,1))	
        z2 = z2_att * z2	

        # z3
        down_2 = self.conv_d2(right)
        if down_2.size()[2:] != left.size()[2:]:
            down_2 = F.interpolate(down_2, size=left.size()[2:], mode="bilinear")
        z3 = F.relu(down_2 * left, inplace=True)  # down_2 is mask

        z3_att = F.adaptive_avg_pool2d(self.conv_att3(z3), (1,1))	
        z3 = z3_att * z3	
        out = (z1 + z2 + z3) / (z1_att + z2_att + z3_att)	
        return F.relu(self.bn3(self.conv3(out)), inplace=True)	

        # out = torch.cat((z1, z2, z3), dim=1)
        # return F.relu(self.bn3(self.conv3(out)), inplace=True)



class SA(nn.Module):
    def __init__(self, in_channel_left, in_channel_down):
        super(SA, self).__init__()
        self.conv0 = nn.Conv2d(in_channel_left, 256, kernel_size=3, stride=1, padding=1)
        self.bn0 = nn.BatchNorm2d(256)
        self.conv2 = nn.Conv2d(in_channel_down, 512, kernel_size=3, stride=1, padding=1)

    def forward(self, left, down):
        left = F.relu(self.bn0(self.conv0(left)), inplace=True)  # 256 channels
        down_1 = self.conv2(down)  # wb
        if down_1.size()[2:] != left.size()[2:]:
            down_1 = F.interpolate(down_1, size=left.size()[2:], mode="bilinear")
        w, b = down_1[:, :256, :, :], down_1[:, 256:, :, :]

        return F.relu(w * left + b, inplace=True)

class GCPAGALDNet(nn.Module):
    def __init__(self):
        super(GCPAGALDNet, self).__init__()
        # self.cfg = cfg
        # self.bkbone = ResNet()
        self.resnet = res2net50_v1b_26w_4s(pretrained=True)

        self.ca45 = CA(2048, 2048)
        self.ca35 = CA(2048, 2048)
        self.ca25 = CA(2048, 2048)
        self.ca55 = CA(256, 2048)
        self.sa55 = SA(2048, 2048)

        self.fam45 = FAM(1024, 256, 256)
        self.fam34 = FAM(512, 256, 256)
        self.fam23 = FAM(256, 256, 256)

        self.srm5 = SRM(256)
        self.srm4 = SRM(256)
        self.srm3 = SRM(256)
        self.srm2 = SRM(256)

        self.linear5 = nn.Conv2d(256, 1, kernel_size=3, stride=1, padding=1)
        self.linear4 = nn.Conv2d(256, 1, kernel_size=3, stride=1, padding=1)
        self.linear3 = nn.Conv2d(256, 1, kernel_size=3, stride=1, padding=1)
        self.linear2 = nn.Conv2d(256, 1, kernel_size=3, stride=1, padding=1)

        inplanes = 2048
        interplanes = 256
        num_classes = 1
        self.conva_gald = nn.Sequential(
            nn.Conv2d(inplanes, interplanes, 3, padding=1, bias=False),
            BatchNorm2d(interplanes),
            nn.ReLU(interplanes),
        )
        self.a2block_gald = GALDBlock(interplanes, interplanes // 2)
        self.convb_gald = nn.Sequential(
            nn.Conv2d(interplanes, interplanes, 3, padding=1, bias=False),
            BatchNorm2d(interplanes),
            nn.ReLU(interplanes),
        )

        self.bottleneck_gald = nn.Sequential(
            nn.Conv2d(
                inplanes + interplanes,
                interplanes,
                kernel_size=3,
                padding=1,
                dilation=1,
                bias=False,
            ),
            BatchNorm2d(interplanes),
            nn.ReLU(interplanes),
            nn.Conv2d(
                interplanes, num_classes, kernel_size=1, stride=1, padding=0, bias=True
            ),
        )

    def forward(self, x):

        out1 = self.resnet.conv1(x)
        out1 = self.resnet.bn1(out1)
        out1 = self.resnet.relu(out1)
        out1 = self.resnet.maxpool(out1)  # bs, 64, 88, 88

        out2 = self.resnet.layer1(out1)  # bs, 256, 88, 88
        out3 = self.resnet.layer2(out2)  # bs, 512, 44, 44
        out4 = self.resnet.layer3(out3)  # bs, 1024, 22, 22
        out5_ = self.resnet.layer4(out4)  # bs, 2048, 11, 11
        # out1, out2, out3, out4, out5_ = self.bkbone(x)
        # GCF
        out4_a = self.ca45(out5_, out5_)  # bs, 256, 11, 11
        out3_a = self.ca35(out5_, out5_)  # bs, 256, 11, 11
        out2_a = self.ca25(out5_, out5_)  # bs, 256, 11, 11

        # HA

        out5__ = self.conva_gald(out5_)
        out5__ = self.a2block_gald(out5__)
        x5_head_out = self.convb_gald(out5__)
        x5_head_out = self.bottleneck_gald(torch.cat([out5_, x5_head_out], 1))
        # print(x5_head_out.shape, "bottleneck_gald4", x5_head_out.shape)
        x5_head_out = F.interpolate(x5_head_out, scale_factor=32, mode="bilinear")

        # out5_a = self.sa55(out5_, out5_)
        # out5 = self.ca55(out5_a, out5_)  # bs, 256, 11, 11

        # out
        out5 = self.srm5(out5__)  # bs, 256, 11, 11
        # out5 = self.srm5(out5)  # bs, 256, 11, 11

        out4 = self.srm4(self.fam45(out4, out5, out4_a))
        out3 = self.srm3(self.fam34(out3, out4, out3_a))
        out2 = self.srm2(self.fam23(out2, out3, out2_a))
        # we use bilinear interpolation instead of transpose convolution
        out5 = F.interpolate(self.linear5(out5), size=x.size()[2:], mode="bilinear")
        out4 = F.interpolate(self.linear4(out4), size=x.size()[2:], mode="bilinear")
        out3 = F.interpolate(self.linear3(out3), size=x.size()[2:], mode="bilinear")
        out2 = F.interpolate(self.linear2(out2), size=x.size()[2:], mode="bilinear")
        return x5_head_out, out5, out4, out3, out2