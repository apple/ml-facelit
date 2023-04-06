# FaceLit: Neural 3D Relightable Faces

This is the official repository of

*Anurag Ranjan, Kwang Moo Yi, Rick Chang, Oncel Tuzel*, **FaceLit: Neural 3D Relightable Faces.** CVPR 2023

[![arxiv](https://shields.io/badge/paper-green?logo=arxiv&style=for-the-badge)](https://arxiv.org/abs/2303.15437)
[![webpage](https://shields.io/badge/Webpage-green?logo=safari&style=for-the-badge)](https://machinelearning.apple.com/research/neural-3d-relightable)


https://user-images.githubusercontent.com/14334441/229917229-ab587c29-7250-46ab-9f42-12b52bb141de.mp4


## Setup
```bash
conda create -f facelit/enviroment.yml
conda activate facelit
```

## Demo

Download pretrained models

```bash
bash download_models.sh
```

Generate video demos.

```bash
python gen_videos.py --outdir=out --trunc=0.7 --seeds=0-3 --grid=2x2 --network=pretrained/NETWORK.pkl --light_cond=True --entangle=[camera, light, lightcam, specular, specularcam]
```

## Training

Train with a neural rendering resolution of 64x64
```bash
python train.py --outdir==out --cfg=ffhq --data=DATA_DIR --gpus=8 --batch=32 --gamma=1 --gen_pose_cond=True --gen_light_cond=True --light_mode=[diffuse, full] --normal_reg_weight=1e-4 --neural_rendering_resolution_final=64
```

Fine tune with a neural rendering resolution of 128x128
```bash
python train.py --outdir==out --cfg=ffhq --data=DATA_DIR --gpus=8 --batch=32 --gamma=1 --gen_pose_cond=True --gen_light_cond=True --light_mode=[diffuse, full] --normal_reg_weight=1e-4 --neural_rendering_resolution_final=128 --resume=pretrained/NETWORK.pkl
```

## Data Preprocessing
We use the dataset from [EG3D](https://github.com/NVlabs/eg3d) and obtain camera parameters and illumination parameters using [DECA](https://github.com/yfeng95/DECA).

#### Setting up DECA

```bash
git clone https://github.com/YadiraF/DECA.git
cd DECA
git checkout 022ed52
bash install_conda.sh
conda activate deca-env
bash fetch_data.sh
```

Apply our patch

```bash
git apply FACELIT_DIR/third_party/deca.patch
```

To generate deca fits, run `generate_deca_fits.sh`.


## Evaluation

Evaluation of models requires setting up DECA ([see here](####Setting-up-DECA)) and setting up Deep3DFaceRecon (see below).

#### Setting up Deep3DFaceRecon
Use this [fork](https://github.com/Xiaoming-Zhao/Deep3DFaceRecon_pytorch) to set up Deep3DFaceRecon_pytorch.
```bash
git clone https://github.com/Xiaoming-Zhao/Deep3DFaceRecon_pytorch
```
 
To run the evaluation, run `eval_metrics.sh`. Note that due to randomness in the generation process, the metrics reported might vary by  ±2%.

## Citation

```
@inproceedings{ranjan2023,
  author = {Anurag Ranjan and Kwang Moo Yi and Rick Chang and Oncel Tuzel},
  title = {FaceLit: Neural 3D Relightable Faces},
  booktitle = {Proceedings of the IEEE conference on computer vision and pattern recognition},
  year = {2023}
}
```

## Acknowledgements

This code is based on [EG3D](https://github.com/NVlabs/eg3d), we thank the authors for their github contribution. We also use portions of the code from  [GMPI](https://github.com/apple/ml-gmpi).
