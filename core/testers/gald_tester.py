from tqdm import tqdm
import numpy as np
import os

import torch

from core.models.classifiers.gcpacc.gcpa_cc2 import GCPADecoder, GCPAEncoder
from core.utils.utility import intersectionAndUnionGPU, AverageMeter, get_color_palette

class GALDTester:
    def __init__(self, cfg, device, test_loader, logger, palette, saveres=False):
        self.cfg = cfg
        self.logger = logger
        self.test_loader = test_loader
        self.device = device
        self.palette = palette
        self.saveres = saveres

        self.encoder = GCPAEncoder()
        self.decoder = GCPADecoder()
        self.encoder.to(device)
        self.decoder.to(device)
    
    def _load_checkpoint(self):
        self.logger.info("Loading checkpoint from {}".format(self.cfg.resume))
        checkpoint = torch.load(self.cfg.resume, map_location=self.device)
        feature_extractor_weights = strip_prefix_if_present(checkpoint['feature_extractor'], 'module.')
        self.feature_extractor.load_state_dict(feature_extractor_weights)
        classifier_weights = strip_prefix_if_present(checkpoint['classifier'], 'module.')
        self.classifier.load_state_dict(classifier_weights)

    def save_distill(self, output, name):
        """
        :param output: tensor (B=1) x C x H x W

        """
        dataset_name = self.cfg.DATASETS.TEST
        output_folder = os.path.join(self.cfg.PSEUDO_DIR, "inference", dataset_name)
        os.makedirs(output_folder)
        output = output.cpu().numpy().squeeze() # numpy array C x H x W
        pred = output.argmax(0) # numpy array H x W
        mask = get_color_palette(pred, self.palette)
        mask_filename = name[0] + '.png'
        mask.save(os.path.join(output_folder, mask_filename))

    def test(self):
        self.encoder.eval()
        self.decoder.eval()

        self.meter = AverageMeter()

        for batch in tqdm(self.test_loader):
            x, y, name = batch
            x = x.cuda(non_blocking=True)
            y = y.cuda(non_blocking=True).long()

            hardnetout = self.encoder(x)
            res5, res4, res3, res2 = self.decoder(x, hardnetout)


            name = os.path.splitext(filename[0])[0]
            ext = os.path.splitext(filename[0])[1]
            gt = gt[0][0]
            gt = np.asarray(gt, np.float32)

            res = res2
            res = F.upsample(
                res, size=gt.shape, mode="bilinear", align_corners=False
            )
            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)

            pred = output.max(1)[1] # tensor B, H, W
            if self.saveres:
                self.save_distill(output, name)
                
            intersection, union, target, res = intersectionAndUnionGPU(pred, y, self.cfg.MODEL.NUM_CLASSES, self.cfg.INPUT.IGNORE_LABEL)
            intersection, union, target, res = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy(), res.cpu().numpy()

            self.meter.update(intersection, union, target, res)

        self.meter.summary(self.logger, self.cfg.MODEL.NUM_CLASSES)