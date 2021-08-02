export CUDA_VISIBLE_DEVICES=1

# python -m torch.distributed.launch --nproc_per_node=1 train_src.py -cfg configs/deeplabv2_r101_src.yaml OUTPUT_DIR results/src_r101_try/

python -m torch.distributed.launch --nproc_per_node=1 train_src.py -cfg configs/gald_src.yaml OUTPUT_DIR results/src_gald

python -m torch.distributed.launch --nproc_per_node=1 train_adv.py -cfg configs/deeplabv2_r101_adv.yaml OUTPUT_DIR results/adv_test resume /mnt/data/taint/weights/Aspp-4.pth

# python test.py -cfg configs/deeplabv2_r101_adv.yaml --saveres OUTPUT_DIR results/adv_test/ PSEUDO_DIR /home/admin_mcn/taint/dataset/cityscapes/soft_labels DATASETS.TEST cityscapes_train resume results/adv_test/model_iter040000.pth

# python demo.py -cfg configs/deeplabv2_r101_adv.yaml