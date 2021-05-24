from tqdm import tqdm
import numpy as np
import os

import torch

from core.models.build import build_feature_extractor, build_classifier
from core.utils.utility import strip_prefix_if_present, inference, multi_scale_inference, intersectionAndUnion, intersectionAndUnionGPU, AverageMeter, get_color_palette, confusion_matrix, plot_confusion_matrix, dump_json

class ASPPTester:
    def __init__(self, cfg, device, test_loader, logger, palette, trainid2name, saveres=False):
        self.cfg = cfg
        self.logger = logger
        self.test_loader = test_loader
        self.device = device
        self.palette = palette
        self.trainid2name = trainid2name
        self.saveres = saveres
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
        num_classes = self.cfg.MODEL.NUM_CLASSES
        self.feature_extractor.eval()
        self.classifier.eval()

        self.meter = AverageMeter()
        cmt = torch.zeros(num_classes, num_classes, dtype=torch.int64)

        for batch in tqdm(self.test_loader):
            x, y, name = batch
            x = x.cuda(non_blocking=True)
            y = y.cuda(non_blocking=True).long()

            output = inference(self.feature_extractor, self.classifier, x, y, flip=False) # tensor B x C x H x W
            # output = multi_scale_inference(self.feature_extractor, self.classifier, x, y, flip=False) # tensor B x C x H x W

            pred = output.max(1)[1] # tensor B, H, W
            if self.saveres:
                self.save_distill(output, name)

            pds = torch.flatten(pred) # vector tensor (B x H x W)
            gts = torch.flatten(y) # vector tensor (B x H x W)
            cmt = cmt + confusion_matrix(self.cfg, pds, gts)
            
            intersection, union, target, res = intersectionAndUnionGPU(pred, y, self.cfg.MODEL.NUM_CLASSES, self.cfg.INPUT.IGNORE_LABEL)
            intersection, union, target, res = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy(), res.cpu().numpy()

            self.meter.update(intersection, union, target, res)

        # plot_confusion_matrix(cmt, list(self.trainid2name.values()))
        self.meter.summary(self.logger, num_classes)
        mydata = {
            'cmt': cmt,
            'classes': list(self.trainid2name.values())
        }
        json_path = os.path.join(self.cfg.OUTPUT_DIR, "aspp_confusion_matrix.json")
        dump_json(json_path, mydata)