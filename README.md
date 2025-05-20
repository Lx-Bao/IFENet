# IFENet

This repo is an official implementation of the *IFENet*.
**IFENet: Interaction, Fusion, and Enhancement network for VDT Salient Object Detection. IEEE Transactions on Image Processing (2025).**


## Prerequisites

## Usage

### 1. Clone the repository

### 2. Training
Download the pretrained model **swin_base_patch4_window12_384_22k.pth**. <br>
Download the edge Ground Truth of the training set of VDT-2048 dataset [here](https://pan.baidu.com/s/1T_zM6msG7e1Xg5bIzaWBxA?pwd=u450) (fetch code: u450) and put it on the training set folder.

You can train the model by using 
```
python Train.py
```

### 3. Testing
```
python Test.py
```

### 4. Evaluation

- We provide [saliency maps](https://pan.baidu.com/s/1Girb29F6WxQzUjNU6jFn7w?pwd=k3qe) (fetch code: k3qe) of our IFENet on VDT-2048 dataset.
- The edge Ground Truth of the training set of VDT-2048 dataset can be download [here](https://pan.baidu.com/s/1T_zM6msG7e1Xg5bIzaWBxA?pwd=u450) (fetch code: u450)


## Citation
```
@article{bao2025ifenet,
  title={IFENet: Interaction, Fusion, and Enhancement network for VDT Salient Object Detection},
  author={Bao, Liuxin and Zhou, Xiaofei and Zheng, Bolun and Cong, Runmin and Yin, Haibing and Zhang, Jiyong and Yan, Chenggang},
  journal={IEEE Transactions on Image Processing},
  year={2025},
  publisher={IEEE}
}
```

- If you have any questions, feel free to contact us via: `lxbao@hdu.edu.cn` or `zxforchid@outlook.com`.
