# Experiments on CelebA

## Datset download

1. Down load a zip file from https://www.kaggle.com/datasets/badasstechie/celebahq-resized-256x256 and rename it to `CelebA_hq_256.zip`
2. unzip CelebA_hq_256.zip and get the dataset folder `CelebA_hq_256`
3. encode the datasets into StyleGAN format by running the following command. The size of the obatined zip is about 5.6G

```
python dataset_tool.py --source CelebA_hq_256 --dest CelebA_hq_256_stylegan_parsed.zip --resolution 256x256
```

## Train the StyleGAN2 on CelebA

1. train the default StyleGAN2
```
python train.py --outdir=./logs --cfg=stylegan2 --data /path/to/celeba_hq_256_stylegan_parsed.zip  \
--gpus 4 --batch=64 --gamma 10 --mirror=1 --aug=noaug
```

2. train our searched StyleGAN2. 



```
python -m ipdb train.py --outdir=./logs --cfg=stylegan2 --data /path/to/celeba_hq_256_stylegan_parsed.zip  \
--gpus 4 --batch=64 --gamma 10 --mirror=1 --aug=noaug --arch /path/to/eagan_celebA.json
```

where `eagan_celebA.json` is 

```
{
    '128': 5,
    '256': 5
}
```

## Evaluate FID and IS

```
python calc_metrics.py --metrics=is50k,fid50k --data=/path/to/celeba_hq_256_stylegan_parsed.zip --mirror=1 \
    --network=/path/to/logs/00010-stylegan2-celeba_hq_256_stylegan_parsed-gpus4-batch64-gamma10/network-snapshot-004636.pkl --gpus 4
```