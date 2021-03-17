import argparse
import os
from collections import OrderedDict

import torch
import torch.nn.functional as F

from core.configs import cfg
from core.datasets.build import build_dataset, build_collate_fn
from core.combos.aspp_fada import AsppFada
from core.combos.attn_fada import AttnFada

def main(name, cfg, local_rank, distributed):
    src_train_data = build_dataset(cfg, mode='train', is_source=True)
    tgt_train_data = build_dataset(cfg, mode='train', is_source=False)
    collate_fn = build_collate_fn(cfg)

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
        collate_fn=collate_fn,
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
    elif name == "attn_fada":
        trainer = AttnFada(name, cfg, src_train_loader, tgt_train_loader, local_rank)
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

    main("attn_fada", cfg, args.local_rank, args.distributed)