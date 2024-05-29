# GAN Inversion for Image Editing via Unsupervised Domain Adaptation
The official PyTorch implementation of GAN Inversion for Image Editing via Unsupervised Domain Adaptation. Our implementation is based on the [HFGI](https://github.com/Tengfei-Wang/HFGI) and [E2style](https://github.com/wty-ustc/e2style) codebase.
## Getting Started

### Installation

```
conda env create -f environment.yaml
conda activate UDA
```

### Pre-trained Models
Please download our pre-trained model trained with FFHQ dataset [here](https://drive.google.com/file/d/1r0_sNKyAqkGRwY0ghyj9mgwYswyFKXKB/view?usp=sharing).

## Training & Inference
###  Dataset Preparation
A generic data loader where the images are arranged in this way by default:
```
ffhq
    test
        xxx.png
        yyy.png
        ...
    train
        src
            zzz.png
            uuu.png    
            ...
        trg
            vvv.png
            www.png
            ...

```
### Training
Modify  `options/train_options.py` and run:
```
python ./scripts/train.py  \
--dataset_type='uda_ffhq_encode'  --start_from_latent_avg \
--val_interval=2000 --save_interval=10000 \
--max_steps=200000  --stylegan_size=1024 \
--is_train=True --distortion_scale=0.0 \
--aug_rate=0.0 --res_lambda=0.0  \
--dst_lambda=0.5 --id_lambda=1.0 \
--checkpoint_path=''  --exp_dir='./experiment/' \
--workers=2  --batch_size=2  --test_batch_size=2 \
--test_workers=2   --board_interval=10
```
### Inference
You can use `scripts/new_infer.py` to apply the model on a set of images.
For example,
```
python ./scripts/new_infer.py \
--images_dir=/path/to/test_data  --n_sample=100 --edit_attribute='inversion'  \
--save_dir=./results  /path/to/pre-trained-model
```

```
python ./scripts/new_infer.py \
--images_dir=/path/to/test_data  --n_sample=100 --edit_attribute='smile' --edit_degree=-1 \
--save_dir=./results  /path/to/pre-trained-model
```

For edit_attribute,	we provide options of 'inversion', 'age', 'smile', 'eyes', 'lip' and 'beard' in the script. Edit_degree	controls the degree of editing (works for 'age' and 'smile').