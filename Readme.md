# NTIRE 2024: Stereo Image Super-Resolution Challenge <br>
<p align="center">  <img src="https://raw.github.com/YingqianWang/Stereo-Image-SR/NTIRE2024/Fig/logo.png" width="50%"> </p>

**Stereo image super-resolution (SR) challenge is held as a part of the [NTIRE workshop](https://cvlai.net/ntire/2024/) in conjunction with CVPR 2024. The goal of this challenge is to develop methods to recover high-resolution (HR) stereo image pairs.** <br>

## News and Updates:
* **2024-01-29**: CodaLab servers for [Track2](https://codalab.lisn.upsaclay.fr/competitions/17246) are online. Training and validation data has been released.

### Track 2: [Constrained SR & Realistic Degradation](https://codalab.lisn.upsaclay.fr/competitions/17246)
#### Degradation Model:
In this track, a realistic degradation model consisting of blur, downsampling, noise, and compression is adopted to synthesize LR images:


$$I^{LR}=\mathcal{C}\left(\left(I^{HR}\otimes{k}\right)\downarrow_s+n\right),$$

where $k$ is the blur kernel, $n$ is additive Gaussian noise, and $\mathcal{C}$ represents JPEG compression.

### Dataset
The dataset was supported by challenge organizers

(1) **Training Set**
The 800 stereo images in training set of the Flickr1024 dataset are used as the training set of this challenge. Both HR images and their LR versions (produced by realistic degradations) can be downloaded via [Jianguo Drive](https://www.jianguoyun.com/p/Da4VFtMQstPqChjpkrsFIAA) or [Google Drive](https://drive.google.com/drive/folders/136h3xlftEIc2NzKmDaw-8DlRoiO1ioOO). The participants can use these HR images as groundtruth to train their models.

(2) **Validation Set**
The 112 stereo images in the validation set of the Flickr1024 dataset are used as the validation set of this challenge. Only LR images (produced by realistic degradations) are provided. The participants can download the validation set via [Jianguo Drive](https://www.jianguoyun.com/p/Da4VFtMQstPqChjpkrsFIAA) or [Google Drive](https://drive.google.com/drive/folders/136h3xlftEIc2NzKmDaw-8DlRoiO1ioOO) to evaluate the performance of their developed models by submitting their super-resolved images to the CodaLab server. Note that, the validation set should be used for validation purpose only but cannot be used as additional training data.

(3) **Test Set**
To rank the submitted models, a test set consisting of 100 stereo images are provided. Only LR images (produced by realistic degradations) are released. The participants are required to apply their models to the released LR stereo images (to be released) and submit their super-resolved images to the server. The participants can download the test set via [Jianguo Drive](https://www.jianguoyun.com/p/Da4VFtMQstPqChjpkrsFIAA) or [Google Drive](https://drive.google.com/drive/folders/136h3xlftEIc2NzKmDaw-8DlRoiO1ioOO). It should be noted that the images in the test set (even the LR versions) cannot be used for training.

## Prepare
The pretrained model has already been placed in the ```./experiments/pretrained_models```

You need to modify the dataset address in the ```./options``` flolder of the ```NAFSSR-T_4x.yml``` file.


## 1. Evaluation

```
python -m torch.distributed.launch --nproc_per_node=1 --master_port=5321 basicsr/test.py -opt options/test/NAFSSR/NAFSSR-T_4x.yml --launcher pytorch
```


## 2. Training


  ```
python -m torch.distributed.launch --nproc_per_node=1 --master_port=4324 basicsr/train.py -opt options/train/NAFSSR/NAFSSR-T_x4.yml --launcher pytorch
  ```
