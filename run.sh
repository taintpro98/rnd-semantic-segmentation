export CUDA_VISIBLE_DEVICES=1

# python -m torch.distributed.launch --nproc_per_node=1 train_src.py -cfg configs/deeplabv2_r101_src.yaml OUTPUT_DIR results/src_r101_merge/

# python test.py -cfg configs/deeplabv2_r101_src.yaml OUTPUT_DIR results/src_r101_test/ resume results/src_r101_merge/Aspp-2.pth


# python -m torch.distributed.launch --nproc_per_node=1 train_src.py -cfg configs/gald_src.yaml OUTPUT_DIR results/src_gald

python -m torch.distributed.launch --nproc_per_node=1 train_adv.py -cfg configs/deeplabv2_r101_adv.yaml OUTPUT_DIR results/adv_test resume results/src_r101_merge/Aspp-2.pth

# python -m torch.distributed.launch --nproc_per_node=1 train_adv.py -cfg configs/gald_adv.yaml OUTPUT_DIR results/adv_gald resume /mnt/data/taint/weights/Gald-8.pth

# python test.py -cfg configs/deeplabv2_r101_adv.yaml --saveres OUTPUT_DIR results/adv_test/ PSEUDO_DIR /home/admin_mcn/taint/dataset/cityscapes/soft_labels DATASETS.TEST cityscapes_train resume results/adv_test/model_iter040000.pth

# python test.py -cfg configs/deeplabv2_r101_tgt_self_distill.yaml OUTPUT_DIR results/sd_test/ resume /mnt/data/taint/weights/Aspp-80.pth

# python test.py -cfg configs/gald_src.yaml OUTPUT_DIR results/src_gald/ resume results/src_gald/Gald-1.pth
# python test.py -cfg configs/deeplabv2_r101_src.yaml OUTPUT_DIR results/src_r101_try/ resume /mnt/data/taint/weights/gta5/gta5-21-4-2021/Aspp-4.pth

# python demo.py -cfg configs/deeplabv2_r101_tgt_self_distill.yaml

# python -m torch.distributed.launch --nproc_per_node=1 train_src.py -cfg configs/deeplabv2_r101_tgt_self_distill.yaml OUTPUT_DIR results/sd_test/
