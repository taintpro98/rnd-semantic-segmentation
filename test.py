import argparse
import logging

import torch

from core.configs import cfg
from core.datasets.build import build_dataset
from core.testers.aspp_tester import ASPPTester
from core.testers.pranet_tester import PranetTester
from core.utils.utility import load_json, setup_logger

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def test(cfg, config):
    name = config["name"]
    logger = setup_logger(name, cfg.OUTPUT_DIR, None, config['filename'])
    logger.info("#"*20 + " Start Testing " + "#"*20)

    test_data = build_dataset(cfg, name, mode='test', is_source=False, epochwise=False)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=cfg.TEST.BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True, sampler=None)

    if name == "aspp":
        tester = ASPPTester(cfg, device, test_loader, logger)
    elif name == "pranet":
        tester = PranetTester(cfg, device, test_loader, logger)
    tester._load_checkpoint()
    tester.test()

def main():
    parser = argparse.ArgumentParser(description="PyTorch Semantic Segmentation Testing")
    parser.add_argument("-cfg",
        "--config-file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument('-c', '--config_path', default='configs/demo_config.json', help='path to config')
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    
    args = parser.parse_args()
    config = load_json(args.config_path)

    torch.backends.cudnn.benchmark = True

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    print("Loaded configuration file {}".format(args.config_file))
    test(cfg, config)

if __name__ == "__main__":
    main()