export CUDA_VISIBLE_DEVICES=1

python -m torch.distributed.launch --nproc_per_node=1 train_src.py -cfg configs/deeplabv2_r101_src_kvasir.yaml OUTPUT_DIR results/src_r101_try/

python -m torch.distributed.launch --nproc_per_node=1 train_adv.py -cfg configs/pranet_adv_polyp_bli.yaml OUTPUT_DIR results/adv_test resume ../../weights/f0-aspp-19-2-2021/model_iter016000.pth

python test.py -cfg configs/attn_adv_kvasir.yaml OUTPUT_DIR results/fada_attn_kvasir2bli/ resume results/src_r101_try/model_iter016000.pth

python demo.py -cfg configs/attn_adv_kvasir.yaml