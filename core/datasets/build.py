from .dataset_path_catalog import DatasetCatalog
from core.components.augment import Augmenter
from core.datasets.func import *

def build_collate_fn(cfg):
    if cfg.AUG.COLLATE == "attn":
        return attn_collate_fn
    elif cfg.AUG.COLLATE == "pranet":
        return pranet_collate_fn
    elif cfg.AUG.COLLATE == None:
        return None
    else:
        return None

def build_dataset(cfg, mode='train', is_source=True):
    assert mode in ['train', 'val', 'test']
    # transform = build_transform(cfg, mode, is_source)
    augmenter = Augmenter(cfg, mode, is_source)
    transform = augmenter.build_transform()
    
    if mode=='train':
        if is_source:
            dataset = DatasetCatalog.get(cfg, cfg.DATASETS.SOURCE_TRAIN, mode, num_classes=cfg.MODEL.NUM_CLASSES, transform=transform, cross_val=cfg.DATASETS.CROSS_VAL)
        else:
            dataset = DatasetCatalog.get(cfg, cfg.DATASETS.TARGET_TRAIN, mode, num_classes=cfg.MODEL.NUM_CLASSES, transform=transform, cross_val=cfg.DATASETS.CROSS_VAL)
    elif mode=='val':
        dataset = DatasetCatalog.get(cfg, cfg.DATASETS.TEST, 'val', num_classes=cfg.MODEL.NUM_CLASSES, transform=transform, cross_val=cfg.DATASETS.CROSS_VAL)
    elif mode=='test':
        dataset = DatasetCatalog.get(cfg, cfg.DATASETS.TEST, cfg.DATASETS.TEST.split('_')[-1], num_classes=cfg.MODEL.NUM_CLASSES, transform=transform, cross_val=cfg.DATASETS.CROSS_VAL)
    return dataset