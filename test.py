import argparse
import logging
from tqdm import tqdm

import torch

from core.utils.utility import strip_prefix_if_present, inference, intersectionAndUnion, intersectionAndUnionGPU

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def test(cfg):
    logger = logging.getLogger("FADA.tester")
    logger.info("Start testing")
    # device = torch.device(cfg.MODEL.DEVICE)

    feature_extractor = build_feature_extractor(cfg)
    feature_extractor.to(device)
    
    classifier = build_classifier(cfg)
    classifier.to(device)

    logger.info("Loading checkpoint from {}".format(cfg.resume))
    checkpoint = torch.load(cfg.resume, map_location=torch.device('cpu'))
    feature_extractor_weights = strip_prefix_if_present(checkpoint['feature_extractor'], 'module.')
    feature_extractor.load_state_dict(feature_extractor_weights)
    classifier_weights = strip_prefix_if_present(checkpoint['classifier'], 'module.')
    classifier.load_state_dict(classifier_weights)

    feature_extractor.eval()
    classifier.eval()

    for batch in tqdm(test_loader):
        x, y, name = batch
        x = x.cuda(non_blocking=True)
        y = y.cuda(non_blocking=True).long()

        pred = inference(feature_extractor, classifier, x, y, flip=False)

        output = pred.max(1)[1]
        intersection, union, target, output = intersectionAndUnionGPU(output, y, cfg.MODEL.NUM_CLASSES, cfg.INPUT.IGNORE_LABEL)
        intersection, union, target, output = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()


def main():
    parser = argparse.ArgumentParser(description="PyTorch Semantic Segmentation Testing")
    parser.add_argument("-cfg",
        "--config-file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument('--saveres', action="store_true",
                        help='save the result')
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    test(cfg)

if __name__ == "__main__":
    main()