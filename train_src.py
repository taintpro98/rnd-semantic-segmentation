import glob
import os
import logging

import torch
import torch.nn.functional as F
# from base_trainer import BaseTrainer

from core.models import build_model, build_feature_extractor, build_classifier

if torch.cuda.is_available():
    self.with_cuda = True
        device = 'cuda'
    else:
        self.logger.warning('Warning: There\'s no CUDA support on this machine, ''training is performed on CPU.')
        self.with_cuda = False
        device = 'cpu'

def train(cfg):
    logger = logging.getLogger("FADA.trainer")
    logger.info("Start training")

    feature_extractor = build_feature_extractor(cfg)
    device = torch.device(cfg.MODEL.DEVICE)
    feature_extractor.to(device)
    
    classifier = build_classifier(cfg)
    classifier.to(device)

    optimizer_fea = torch.optim.SGD(feature_extractor.parameters(), lr=cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM, weight_decay=cfg.SOLVER.WEIGHT_DECAY)
    optimizer_fea.zero_grad()
    
    optimizer_cls = torch.optim.SGD(classifier.parameters(), lr=cfg.SOLVER.BASE_LR*10, momentum=cfg.SOLVER.MOMENTUM, weight_decay=cfg.SOLVER.WEIGHT_DECAY)
    optimizer_cls.zero_grad()

    if cfg.resume:
        logger.info("Loading checkpoint from {}".format(cfg.resume))
        checkpoint = torch.load(cfg.resume, map_location=torch.device('cpu'))
        model_weights = checkpoint['feature_extractor'] if distributed else strip_prefix_if_present(checkpoint['feature_extractor'], 'module.')
        feature_extractor.load_state_dict(model_weights)
        classifier_weights = checkpoint['classifier'] if distributed else strip_prefix_if_present(checkpoint['classifier'], 'module.')
        classifier.load_state_dict(classifier_weights)
        if "optimizer_fea" in checkpoint:
            logger.info("Loading optimizer_fea from {}".format(cfg.resume))
            optimizer.load(checkpoint['optimizer_fea'])
        if "optimizer_cls" in checkpoint:
            logger.info("Loading optimizer_cls from {}".format(cfg.resume))
            optimizer.load(checkpoint['optimizer_cls'])
        if "iteration" in checkpoint:
            iteration = checkpoint['iteration']

    

if __name__ == "__main__":
    main()


