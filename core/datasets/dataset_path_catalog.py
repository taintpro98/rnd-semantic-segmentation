import os
from .cityscapes import cityscapesDataSet, cityscapesSelfDistillDataSet
# from .cityscapes_self_distill import cityscapesSelfDistillDataSet
# from .synthia import synthiaDataSet
from .gta5 import GTA5FoldDataSet
from .gta5_fda import GTA5FDAFoldDataSet
from .carla import CarlaFoldDataSet
from .merge import MergeDataSet
from .kvasir import KvasirDataSet, KvasirFoldDataset
from .bli import BLIDataset

class DatasetCatalog(object):
    DATASETS = {
        "gta5_train": {
            "data_dir": "gta5",
            "data_list": "gta5_train_list.txt"
        },
        "gta5_fda_train": {
            "data_dir": "gta5",
            "data_list": "gta5_train_list.txt"
        },
        "gta5_val": {
            "data_dir": "gta5",
            "data_list": "gta5_train_list.txt"
        },
        "synthia_train": {
            "data_dir": "synthia",
            "data_list": "synthia_train_list.txt"
        },
        "cityscapes_train": {
            "data_dir": "cityscapes",
            "data_list": "cityscapes_train_list.txt"
        },
        "cityscapes_self_distill_train": {
            "data_dir": "cityscapes",
            "data_list": "cityscapes_train_list.txt",
            "label_dir": "cityscapes/soft_labels_3006/inference/cityscapes_train"
        },
        "cityscapes_val": {
            "data_dir": "cityscapes",
            "data_list": "cityscapes_val_list.txt"
        },
        "kvasir_train": {
            "data_dir": "kvasir",
            "data_list": ""
        },
        "kvasir_val": {
            "data_dir": "kvasir",
            "data_list": ""
        },
        "polyp_train": {
            "data_dir": "kvasir",
            "data_list": ""
        },
        "polyp_val": {
            "data_dir": "kvasir",
            "data_list": ""
        },
        "bli_train": {
            "data_dir": "BLI/train",
            "data_list": ""
        },
        "bli_val": {
            "data_dir": "BLI/test",
            "data_list": ""
        },
        "carla_train": {
            "data_dir": "carla",
            "data_list": "carla_train_list.txt"
        },
        "merge_train": {
            "data_dir": "dataset",
            "data_list": "merge_train_list.txt"
        }

    }

    @staticmethod
    def get(cfg, name, mode, num_classes, transform=None, cross_val=None):
        if "gta5" in name:
            data_dir = cfg.DATASETS.DATASET_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                root=os.path.join(data_dir, attrs["data_dir"]),
                data_list=os.path.join(data_dir, attrs["data_list"]),
            )
            if "fda" in name:
                return GTA5FDAFoldDataSet(cfg, args["root"], mode=mode, cross_val=cross_val, transform=transform)

            return GTA5FoldDataSet(cfg, args["root"], mode=mode, cross_val=cross_val, transform=transform)
        elif "synthia" in name:
            data_dir = cfg.DATASETS.DATASET_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                root=os.path.join(data_dir, attrs["data_dir"]),
                data_list=os.path.join(data_dir, attrs["data_list"]),
            )
            return synthiaDataSet(args["root"], args["data_list"], max_iters=max_iters, num_classes=num_classes, split=mode, transform=transform)
        elif "cityscapes" in name:
            data_dir = cfg.DATASETS.DATASET_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                root=os.path.join(data_dir, attrs["data_dir"]),
                data_list=os.path.join(data_dir, attrs["data_list"]),
            )
            if 'distill' in name:
                args['label_dir'] = os.path.join(data_dir, attrs["label_dir"])
                return cityscapesSelfDistillDataSet(args["root"], args['label_dir'], num_classes=num_classes, mode=mode, transform=transform)
            return cityscapesDataSet(args["root"], num_classes=num_classes, mode=mode, transform=transform)
        elif "kvasir" in name:
            data_dir = cfg.DATASETS.DATASET_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                root=os.path.join(data_dir, attrs["data_dir"]),
                data_list=os.path.join(data_dir, attrs["data_list"]),
            )
            return KvasirDataSet(args["root"], num_classes=num_classes, mode=mode, cross_val=cross_val, transform=transform)
        elif "polyp" in name:
            data_dir = cfg.DATASETS.DATASET_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                root=os.path.join(data_dir, attrs["data_dir"]),
                data_list=os.path.join(data_dir, attrs["data_list"]),
            )
            return KvasirFoldDataset(cfg, args["root"], mode=mode, cross_val=cross_val, transform=transform)
        elif "bli" in name:
            data_dir = cfg.DATASETS.DATASET_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                root=os.path.join(data_dir, attrs["data_dir"]),
                data_list=os.path.join(data_dir, attrs["data_list"]),
            )
            return BLIDataset(cfg, args["root"], mode=mode, transform=transform)
        elif "carla" in name:
            data_dir = cfg.DATASETS.DATASET_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                root=os.path.join(data_dir, attrs["data_dir"]),
                data_list=os.path.join(data_dir, attrs["data_list"]),
            )
            return CarlaFoldDataSet(cfg, args["root"], mode=mode, cross_val=cross_val, transform=transform)
        elif "merge" in name:
            data_dir = cfg.DATASETS.DATASET_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                root=data_dir,
                data_list = os.path.join(data_dir, attrs["data_list"]),
            )
            return MergeDataSet(cfg, args["root"], mode=mode, cross_val=cross_val, transform=transform)

        raise RuntimeError("Dataset not available: {}".format(name))
