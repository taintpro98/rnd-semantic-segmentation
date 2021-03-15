import argparse
import logging

import torch

from core.configs import cfg
from core.datasets.build import build_dataset, build_collate_fn
from core.testers.aspp_tester import ASPPTester
from core.testers.pranet_tester import PranetTester
from core.testers.attn_tester import AttnTester
from core.utils.utility import load_json, setup_logger

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def test(cfg, config):
    name = config["name"]
    logger = setup_logger(name, cfg.OUTPUT_DIR, None)
    logger.info("#"*20 + " Start Testing " + "#"*20)

    test_data = build_dataset(cfg, mode='test', is_source=False)
    collate_fn = build_collate_fn(cfg)
    test_loader = torch.utils.data.DataLoader(
        test_data, 
        batch_size=cfg.TEST.BATCH_SIZE, 
        shuffle=False, 
        num_workers=4, 
        pin_memory=True,
        collate_fn=collate_fn, 
        sampler=None
    )

    if name == "aspp_test":
        tester = ASPPTester(cfg, device, test_loader, logger)
    elif name == "pranet_test":
        tester = PranetTester(cfg, device, test_loader, logger)
    elif name == "attn_test":
        tester = AttnTester(cfg, device, test_loader, logger)
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