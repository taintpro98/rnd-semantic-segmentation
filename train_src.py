import argparse

import torch

from core.configs import cfg

from core.trainers.aspp_trainer import ASPPTrainer
from core.trainers.pranet_trainer import PraNetTrainer
from core.trainers.attn_trainer import AttnTrainer
from core.datasets.build import build_dataset, build_collate_fn

def main(name, cfg, local_rank):
    src_train_data = build_dataset(cfg, mode='train', is_source=True)
    collate_fn = build_collate_fn(cfg)
    train_loader = torch.utils.data.DataLoader(
        src_train_data, 
        batch_size=cfg.SOLVER.BATCH_SIZE, 
        shuffle=True, 
        num_workers=4, 
        pin_memory=True, 
        collate_fn=collate_fn,
        sampler=None, 
        drop_last=True
    )

    if name == "aspp":
        trainer = ASPPTrainer(name, cfg, train_loader, local_rank)
    elif name == "pranet":
        trainer = PraNetTrainer(name, cfg, train_loader, local_rank)
    elif name == "attn":
        trainer = AttnTrainer(name, cfg, train_loader, local_rank)
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

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    main("aspp", cfg, args.local_rank)