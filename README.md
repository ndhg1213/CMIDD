# CMIDD

This is the official implementation of paper **Going Beyond Feature Similarity: Effective Dataset distillation based on Class-aware Conditional Mutual Information (ICLR2025)** .

The repository is based on [DC-DSA-DM](https://github.com/VICO-UoE/DatasetCondensation), [MTT](https://github.com/georgecazenavette/mtt-distillation), [IDM](https://github.com/uitrbn/IDM/tree/main), [IDC](https://github.com/snu-mllab/efficient-dataset-condensation) and [IID](https://github.com/VincenDen/IID). Please cite their papers if you use the code. 

### Prepare pretrain models
```
cd pretrain
python pre_model.py
```
### Basic experiments 

```
cd DC-DSA-DM
python main_DM.py  --dataset CIFAR10  --ipc 10 

python main.py  --dataset CIFAR10 --ipc 10  --method DSA  --dsa_strategy color_crop_cutout_flip_scale_rotate
```

```
cd IDC
python condense.py --reproduce  -d cifar10 -f 2 --ipc 10
```


If you use the repo, please consider citing:
```
@article{zhong2024going,
  title={Going Beyond Feature Similarity: Effective Dataset distillation based on Class-aware Conditional Mutual Information},
  author={Zhong, Xinhao and Chen, Bin and Fang, Hao and Gu, Xulin and Xia, Shu-Tao and Yang, En-Hui},
  journal={arXiv preprint arXiv:2412.09945},
  year={2024}
}
```