# Dynamic Neural Surfaces for Elastic 4D Shape Representation and Analysis

A neural framework for the statistical analysis of genus-zero 4D surfaces that deform and evolve overtime. We introduce Dynamic Spherical Neural Surfaces (D-SNS), an efficient and continuous spatiotemporal representation to demonstrate core 4D shape analysis such as spatiotemporal registration, geodesics computation and mean 4D shape estimation framework.

Code:

> __Dynamic Neural Surfaces for Elastic 4D Shape Representation and Analysis__

:rocket: [Project page](https://4d-dsns.github.io/DSNS/)

## Installation

The code is tested on `python=3.11`, as well as `pytorch=2.5` and `torchvision=0.20`. Please follow the instructions [here](https://pytorch.org/get-started/locally/) to install both PyTorch and TorchVision dependencies. Installing both PyTorch and TorchVision with CUDA support is strongly recommended.

The code also require MATLAB for visualization and computation. Please follow the instructions [here](https://au.mathworks.com/help/install/ug/install-products-with-internet-connection.html) to install MATLAB.

We recommend to use conda for installation of all the dependencies. Please follow the command to download the dependencies.

```
conda env create -f environment.yml
```

## Getting started

Please download the [_pretrained](https://drive.google.com/drive/folders/14guvADE9z72j-YaassNjKw_c5hY9lyIm?usp=drive_link) and [data](https://drive.google.com/drive/folders/1jBzhAmneOBCWvWzL6VqFM6JEQ2xRtaVF?usp=drive_link) folder from the google drive. Please put folder structure in the `DSNS-code`:

```
DSNS-code\_pretrained\
DSNS-code\data\
```

### Training

1. Dynamic Spherical Neural Surfaces (D-SNS).
```
python train_dsns.py
```
2. For spatial registration we use Laga et al. codebase please follow the [link](https://github.com/hamidlaga/SRNF).
```
TBA
```
3. Spatiotemporal registration.
```
python train_time_warp.py
```
4. 4D Geodesics.
```
TBA
```
5. Co-registration and 4D mean estimation.
```
TBA
```
### Notebook

TBA (We plan to release the Jupyter Notebook to run the entire neural framework.)

[//]: # (We also provide a demo notebook that shows the entire neural framework. Please note that the `Jupyter Notebook` uses the pretrained D-SNS model and showcases the entire framework. It assumes that the `DSNS-code\data` folder is present in the main directory.)

### Visualization

We showcase the quality of DSNS as heatmap visualization. Similarly, we also highlight spatiotemporal registration visualizations, which are enabled by default. After model training, the visualization will be shown.

## License
This work is made available under the MIT [license](https://github.com/4D-DSNS/DSNS-code/blob/main/LICENSE).