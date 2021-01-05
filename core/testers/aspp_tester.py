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

        intersection_sum = 0
        union_sum = 0
        target_sum = 0
        res_sum = 0
        count = 0
        iou_sum = 0
        f1_sum = 0

        for batch in tqdm(self.test_loader):
            x, y, name = batch
            x = x.cuda(non_blocking=True)
            y = y.cuda(non_blocking=True).long()

            output = inference(self.feature_extractor, self.classifier, x, y, flip=False) # tensor B x C x H x W

            pred = output.max(1)[1] # tensor B, H, W
            intersection, union, target, res = intersectionAndUnionGPU(pred, y, self.cfg.MODEL.NUM_CLASSES, self.cfg.INPUT.IGNORE_LABEL)
            intersection, union, target, res = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy(), res.cpu().numpy()

            iou = intersection/(union + 1e-10)
            f1 = 2*intersection/(target + union + 1e-10)

            count += 1
            intersection_sum += intersection
            union_sum += union
            target_sum += target
            res_sum += res
            iou_sum += iou
            f1_sum += f1

        macro_f1 = f1_sum/float(count)
        macro_iou = iou_sum/float(count)

        micro_f1 = 2*intersection_sum/(target_sum + union_sum + 1e-10)
        micro_iou = intersection_sum/(union_sum + 1e-10)

        mIoU = np.mean(macro_iou)
        mF1 = np.mean(macro_f1)
        self.logger.info('Macro metric, val result: mIoU/mF1 {:.4f}/{:.4f}.'.format(mIoU, mF1))

        mIoU = np.mean(micro_iou)
        mF1 = np.mean(micro_f1)
        self.logger.info('Micro metric, val result: mIoU/mF1 {:.4f}/{:.4f}.'.format(mIoU, mF1))

        for i in range(self.cfg.MODEL.NUM_CLASSES):
            self.logger.info('Macro metric, class {} iou/f1 score: {:.4f}/{:.4f}.'.format(i, macro_iou[i], macro_f1[i]))
            self.logger.info('Micro metric, class {} iou/f1 score: {:.4f}/{:.4f}.'.format(i, micro_iou[i], micro_f1[i]))