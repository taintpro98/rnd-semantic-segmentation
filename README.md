# rnd-semantic-segmentation

## Pretrained models
[You can click here to download Resnet101 weight](https://drive.google.com/file/d/1VHyOJA4YnLnta4Ni4nBdoLaB2W-M-WQ6/view?usp=sharing)

[You can click here to download Hardnet68 weight](https://drive.google.com/file/d/1Pjp7pBqad07_keKrmfsCELNwvnh6btEJ/view?usp=sharing)

## How to run ?

### Install library

### Step-by-step installation

```bash
#dump environment to a file by running this script
conda env export | grep -v "^prefix: " > environment.yml


# install conda environment by running this script 
conda env create -f environment.yml

# follow PyTorch installation in https://pytorch.org/get-started/locally/
# we give the instructions for CUDA 9.2
conda install pytorch==1.2.0 torchvision==0.4.0 cudatoolkit=9.2 -c pytorch
```

### Training

```sh
bash run.sh
```

### Testing

```sh

python test.py -cfg configs/deeplabv2_r101_src.yaml OUTPUT_DIR results/src_r101_try/ resume results/ad_test/Aspp-4.pth
```
<!-- ### [Dataloader](dataloader) (dataloader. name)

All data must put in ./data, follow the tree:

```bash
.
├── Kvasir_SEG_Training_880
│   ├── images
│   │   ├── cju0qkwl35piu0993l0dewei2.png
│   │   ├── cju0qoxqj9q6s0835b43399p4.png
│   └── masks
│       ├── cju0qkwl35piu0993l0dewei2.png
│       ├── cju0qoxqj9q6s0835b43399p4.png
└── Kvasir_SEG_Validation_120
    ├── images
    │   ├── cju0s690hkp960855tjuaqvv0.png
    │   ├── cju0sr5ghl0nd08789uzf1raf.png
    └── masks
        ├── cju0s690hkp960855tjuaqvv0.png
        ├── cju0sr5ghl0nd08789uzf1raf.png
``` -->