from tqdm import tqdm
import numpy as np

import torch
import torch.nn.functional as F

from core.models.classifiers.pranet.PraNet_Res2Net import PraNet
from core.utils.utility import inference, intersectionAndUnion, intersectionAndUnionGPU, AverageMeter

class PranetTester:
    def __init__(self, cfg, device, test_loader, logger):
        self.cfg = cfg
        self.logger = logger
        self.test_loader = test_loader
        self.device = device
        self.model = PraNet()
        self.model.to(device)

    def _load_checkpoint(self):
        self.logger.info("Loading checkpoint from {}".format(self.cfg.resume))
        checkpoint = torch.load(self.cfg.resume, map_location=self.device)
        self.model.load_state_dict(checkpoint["model"])

    def test(self):
        self.model.eval()
        self.meter = AverageMeter()

        for batch in tqdm(self.test_loader):
            x, y, name = batch
            x = x.cuda(non_blocking=True)
            y = y.cuda(non_blocking=True) # tensor B x C X H x W
            # y = np.asarray(y, np.float32)
            # y /= (y.max() + 1e-8)
            _, _, h, w = y.size()
            y = y.squeeze(1)

            res5, res4, res3, res2 = self.model(x)
            output = res2
            output = F.upsample(output, size=(h, w), mode='bilinear', align_corners=False) # tensor B, (C=1), H, W
            output = output.sigmoid().data.cpu().numpy().squeeze(1) # numpy array B, H, W
            output = (output - output.min()) / (output.max() - output.min() + 1e-8) # numpy array B, H, W
            # pred = output.round()
            
            output = np.stack([1-output, output], axis=3) # create a B x H x W x 2 numpy array with 0 background and 1 polyp
            output = torch.from_numpy(output.transpose(0, 3, 1, 2)) # tensor B x 2 x H x W
            pred = output.max(1)[1] # int tensor B, H, W
            pred = pred.cuda(non_blocking=True)

            intersection, union, target, res = intersectionAndUnionGPU(pred, y, self.cfg.MODEL.NUM_CLASSES, self.cfg.INPUT.IGNORE_LABEL)
            intersection, union, target, res = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy(), res.cpu().numpy()

            self.meter.update(intersection, union, target, res)

        self.meter.summary(self.logger, self.cfg.MODEL.NUM_CLASSES)