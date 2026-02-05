# FERMam
FERMam: A Lightweight Dual-Source and Multi-Scale Fusion Framework for Facial Expression Recognition

The purpose of this repository is to support research transparency and reproducibility by releasing the complete source code, training and evaluation scripts, and detailed usage instructions.



## 1. Repository Structure
```
FERMam-main/
├── checkpoint/ # Checkpoint directory (empty by default)
│ └── save_vkp_here
├── data_preprocessing/ # Dataset loading and preprocessing
│ ├── custom_multiprocess.py
│ ├── data_loader.py
│ ├── dataset_affectnet.py
│ ├── dataset_affectnet_8class.py
│ ├── dataset_ferplus.py
│ ├── dataset_raf.py
│ ├── datasets.py
│ ├── image_utils.py
│ ├── plot_confusion_matrix.py
│ └── sam.py
├── models/ # Model definitions
│ ├── pretrain/
│ ├── ASFR.py
│ ├── emotion_hyp.py
│ ├── emotion_hyp_affect.py
│ ├── ir50.py
│ ├── IR_MambaBackbone.py
│ ├── mobilefacenet.py
│ ├── pyramid_mamba_fusion_block.py
│ └── ss2d.py
├── torchsampler/ # Sampling utilities
│ └── FPS.py
├── train.py # Training script (RAF-DB)
├── train_affect.py # Training script (AffectNet)
├── train_FERPlus.py # Training script (FERPlus)
├── test.py # Evaluation script (RAF-DB)
├── test_affect.py # Evaluation script (AffectNet)
├── utils.py # Utility functions
└── README.md
```


> **Note:**  
> Due to dataset license restrictions, the raw datasets are **not included** in this repository.



## 2. Environment Setup

### 2.1 Tested Environment

The code has been tested under the following environment:
```
- Python ≥ 3.8  
- **PyTorch 2.3**
- **CUDA 11.8**
- torchvision  
- numpy  
- opencv-python  
- tqdm  
- einops  
```
### 2.2 Installation

Install dependencies via:

pip install -r requirements.txt

---

## 3. Datasets

FERMam is evaluated on three widely used FER benchmarks.

### 3.1 RAF-DB
```
Official website: http://www.whdeng.cn/RAF/model1.html

Use the basic expression subset

Follow the official train/test split
```
### 3.2 AffectNet
```
Official website: http://mohammadmahoor.com/affectnet/

Use the 7-class or 8-class setting as described in the paper
```
### 3.3 FERPlus
```
Official repository: https://github.com/Microsoft/FERPlus

Follow the official labels and splits
```
---

## 4. Training

### 4.1 Training on RAF-DB
```
python train.py
```
### 4.2 Training on AffectNet
```
python train_affect.py
```
### 4.3 Training on FERPlus
```
python train_FERPlus.py
```
