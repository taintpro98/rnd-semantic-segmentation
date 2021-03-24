from tqdm import tqdm
import numpy as np

import torch

from core.models.build import build_feature_extractor, build_classifier
from core.utils.utility import strip_prefix_if_present, inference, intersectionAndUnion, intersectionAndUnionGPU, AverageMeter

class ASPPTester:
    def __init__(self, cfg, device, test_loader, logger):
        self.cfg = cfg
        self.logger = logger
        self.test_loader = test_loader
        self.device = device
        self.feature_extractor = build_feature_extractor(cfg)
        self.feature_extractor.to(device)
    
        self.classifier = build_classifier(cfg)
        self.classifier.to(device)

    def _load_checkpoint(self):
        self.logger.info("Loading checkpoint from {}".format(self.cfg.resume))
        checkpoint = torch.load(self.cfg.resume, map_location=self.device)
        feature_extractor_weights = strip_prefix_if_present(checkpoint['feature_extractor'], 'module.')
        self.feature_extractor.load_state_dict(feature_extractor_weights)
        classifier_weights = strip_prefix_if_present(checkpoint['classifier'], 'module.')
        self.classifier.load_state_dict(classifier_weights)

    def test(self):
        self.feature_extractor.eval()
        self.classifier.eval()

        self.meter = AverageMeter()

        for batch in tqdm(self.test_loader):
            x, y, name = batch
            x = x.cuda(non_blocking=True)
            y = y.cuda(non_blocking=True).long()

            output = inference(self.feature_extractor, self.classifier, x, y, flip=False) # tensor B x C x H x W

            pred = output.max(1)[1] # tensor B, H, W
            intersection, union, target, res = intersectionAndUnionGPU(pred, y, self.cfg.MODEL.NUM_CLASSES, self.cfg.INPUT.IGNORE_LABEL)
            intersection, union, target, res = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy(), res.cpu().numpy()

            self.meter.update(intersection, union, target, res)

        self.meter.summary(self.logger, self.cfg.MODEL.NUM_CLASSES)