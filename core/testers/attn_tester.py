from tqdm import tqdm
import numpy as np

import torch
import torch.nn.functional as F

from core.models.classifiers.attn.eff import Encoder, Decoder
from core.utils.utility import probs_to_mask, intersectionAndUnionGPU, AverageMeter

class AttnTester:
    def __init__(self, cfg, device, test_loader, logger):
        self.cfg = cfg
        self.logger = logger
        self.test_loader = test_loader
        self.device = device
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.encoder.to(device)
        self.decoder.to(device)

    def _load_checkpoint(self):
        self.logger.info("loading checkpoint from {}".format(self.cfg.resume))
        checkpoint = torch.load(self.cfg.resume, map_location=self.device)
        self.encoder.load_state_dict(checkpoint['encoder'])
        self.decoder.load_state_dict(checkpoint['decoder'])

    def test(self):
        self.encoder.eval()
        self.decoder.eval()

        self.meter = AverageMeter()

        for batch in tqdm(self.test_loader):
            x, y, name = batch
            x = x.cuda(non_blocking=True)
            y = y.cuda(non_blocking=True).long()
            y = y.squeeze(1)

            endpoints = self.encoder(x)
            outputs = self.decoder(endpoints) # [B x C x H x W]
            output = outputs[0]
            pred_probs = torch.sigmoid(output)  # B x C x H x W
            pred = probs_to_mask(pred_probs) # B, H, W

            intersection, union, target, res = intersectionAndUnionGPU(pred, y, self.cfg.MODEL.NUM_CLASSES, self.cfg.INPUT.IGNORE_LABEL)
            intersection, union, target, res = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy(), res.cpu().numpy()

            self.meter.update(intersection, union, target, res)
        
        self.meter.summary(self.logger, self.cfg.MODEL.NUM_CLASSES)