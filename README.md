# IFENet

This repo is an official implementation of the *IFENet*.
**IFENet: Interaction, Fusion, and Enhancement network for VDT Salient Object Detection. IEEE Transactions on Image Processing (2025).**


## Prerequisites

## Usage

### 1. Clone the repository

### 2. Training
Download the pretrained model **swin_base_patch4_window12_384_22k.pth**. <br>

You can train the model by using 
```
python Train.py
```

### 3. Testing
```
python Test.py
```

### 4. Evaluation

- We provide [saliency maps](https://pan.baidu.com/s/1z7kXOXtg1J_lhB1ZNjsTPA?pwd=z9dm) (fetch code: z9dm) of our IFENet on VDT-2048 dataset.


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
