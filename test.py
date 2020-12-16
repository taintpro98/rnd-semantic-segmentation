import argparse
import logging
from tqdm import tqdm

import torch

from core.utils.utility import strip_prefix_if_present, inference, intersectionAndUnion, intersectionAndUnionGPU, AverageMeter
from core.models import build_model, build_adversarial_discriminator, build_feature_extractor, build_classifier
from core.configs import cfg
from core.datasets import build_dataset

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

    test_data = build_dataset(cfg, mode='test', is_source=False)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=cfg.TEST.BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True, sampler=None)

    intersection_sum = 0
    union_sum = 0
    target_sum = 0
    res_sum = 0
    count = 0
    iou_sum = 0
    f1_sum = 0

    for batch in tqdm(test_loader):
        x, y, name = batch
        x = x.cuda(non_blocking=True)
        y = y.cuda(non_blocking=True).long()

        pred = inference(feature_extractor, classifier, x, y, flip=False)

        output = pred.max(1)[1]
        intersection, union, target, res = intersectionAndUnionGPU(output, y, cfg.MODEL.NUM_CLASSES, cfg.INPUT.IGNORE_LABEL)
        intersection, union, target, res = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy(), res.numpy()

        iou = intersection/float(union)
        f1 = 2*intersection/float(target + union)

        count += 1
        intersection_sum += intersection
        union_sum += union
        target_sum += target
        res_sum += res
        iou_sum += iou
        f1_sum += f1

    pp1_f1 = f1_sum/float(count)
    pp1_iou = iou_sum/float(count)

    pp2_f1 = 2*intersection_sum/float(target_sum + union_sum)
    pp2_iou = intersection_sum/float(union_sum)

    mIoU = np.mean(pp1_iou)
    mF1 = np.mean(pp1_f1)
    logger.info('Method 1, val result: mIoU/mF1 {:.4f}/{:.4f}/{:.4f}.'.format(mIoU, mF1))

    mIoU = np.mean(pp2_iou)
    mF1 = np.mean(pp2_f1)
    logger.info('Method 2, val result: mIoU/mF1 {:.4f}/{:.4f}/{:.4f}.'.format(mIoU, mF1))

    for i in range(cfg.MODEL.NUM_CLASSES):
        logger.info('Method 1, class {} iou/f1 score: {:.4f}/{:.4f}.'.format(i, pp1_iou[i], pp1_f1[i]))
        logger.info('Method 2, class {} iou/f1 score: {:.4f}/{:.4f}.'.format(i, pp2_iou[i], pp2_f1[i]))

def main():
    parser = argparse.ArgumentParser(description="PyTorch Semantic Segmentation Testing")
    parser.add_argument("-cfg",
        "--config-file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    torch.backends.cudnn.benchmark = True

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    logger.info("Loaded configuration file {}".format(args.config_file))
    test(cfg)

if __name__ == "__main__":
    main()