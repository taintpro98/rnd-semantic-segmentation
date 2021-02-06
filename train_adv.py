import argparse
import os
from collections import OrderedDict

import torch
import torch.nn.functional as F

from core.configs import cfg
from core.datasets.build import build_dataset
from core.adapters.aspp_fada import AsppFada

def strip_prefix_if_present(state_dict, prefix):
    keys = sorted(state_dict.keys())
    if not all(key.startswith(prefix) for key in keys):
        return state_dict
    stripped_state_dict = OrderedDict()
    for key, value in state_dict.items():
        stripped_state_dict[key.replace(prefix, "")] = value
    return stripped_state_dict

def soft_label_cross_entropy(pred, soft_label, pixel_weights=None):
    N, C, H, W = pred.shape
    loss = -soft_label.float()*F.log_softmax(pred, dim=1)
    if pixel_weights is None:
        return torch.mean(torch.sum(loss, dim=1))
    return torch.mean(pixel_weights*torch.sum(loss, dim=1))

def main(name, cfg, local_rank, distributed):
    src_train_data = build_dataset(cfg, mode='train', is_source=True)
    tgt_train_data = build_dataset(cfg, mode='train', is_source=False)

    if distributed:
        src_train_sampler = torch.utils.data.distributed.DistributedSampler(src_train_data)
        tgt_train_sampler = torch.utils.data.distributed.DistributedSampler(tgt_train_data)
    else:
        src_train_sampler = None
        tgt_train_sampler = None
    
    src_train_loader = torch.utils.data.DataLoader(
        src_train_data, 
        batch_size=cfg.SOLVER.BATCH_SIZE, 
        shuffle=(src_train_sampler is None), 
        num_workers=4, 
        pin_memory=True, 
        sampler=src_train_sampler, 
        drop_last=True
    )
    tgt_train_loader = torch.utils.data.DataLoader(
        tgt_train_data, 
        batch_size=cfg.SOLVER.BATCH_SIZE, 
        shuffle=(tgt_train_sampler is None), 
        num_workers=4, 
        pin_memory=True, 
        sampler=tgt_train_sampler, 
        drop_last=True
    )
    if name == "aspp_fada":
        trainer = AsppFada(name, cfg, src_train_loader, tgt_train_loader, local_rank)
    elif name == "pranet_fada":
        trainer = PraNetFada()
    trainer.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch Semantic Segmentation Training")
    parser.add_argument("-cfg",
        "--config-file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    torch.backends.cudnn.benchmark = True

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = num_gpus > 1

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    # output_dir = cfg.OUTPUT_DIR

    # logger = setup_logger("FADA", output_dir, args.local_rank)
    # logger.info("Using {} GPUs".format(num_gpus))
    # logger.info(args)

    # logger.info("Loaded configuration file {}".format(args.config_file))
    # with open(args.config_file, "r") as cf:
    #     config_str = "\n" + cf.read()
    #     logger.info(config_str)
    # logger.info("Running with config:\n{}".format(cfg))

    main("aspp_fada", cfg, args.local_rank, args.distributed)