# Installation Tutorial

We provide step-by-step installation instructions to create the corresponding environment for using 3D UX-Net and demonstrates the corresponding folder structure to input data samples.


## Conda Environment Setup
Create your own conda environment 
```
conda create -n uxnet3d python=3.8
conda activate uxnet3d
```

Install [Pytorch](https://pytorch.org/) == 1.12.1, [torchvision](https://pytorch.org/vision/stable/index.html) == 0.13.1, cudatooltookit == 11.6.0 (depends on your NVIDIA driver and you can see your compatible CUDA version at the right hand corner in nvidia-smi)
```
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
```
Install [monai](https://github.com/Project-MONAI/MONAI) == 0.9.0
```
pip install monai==0.9.0
```
Clone this repository and install other required packages:
```
git clone git@github.com:MASILab/3DUX-Net.git
pip install -r requirements.txt
```

## Input Folder Format
We initially divide different datasets in the following structure:

    path to all data directory/
    ├── FeTA2021
    ├── FLARE2021
    ├── AMOS2022
    ├── BTCV2015
    ├── LiTS2017
    ├── ...

We further sub-divide the samples into training, validation and testing as follow:

    root_dir/
    ├── imagesTr
    ├── labelsTr
    ├── imagesVal
    ├── labelsVal
    ├── imagesTs
For the input of both training and inference, our code currently only allows data samples in Nifti format. Feel free to provide suggestions on adapting other data formats for both training and inference.
