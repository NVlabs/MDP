# MDP: Multi-Dimensional Pruning for ResNet50

This repository contains the implementation of **Multi-Dimensional Pruning (MDP)** for ResNet50 on ImageNet, including comprehensive ablation studies. Our method combines multi-granularity pruning with advanced latency modeling to achieve optimal latency-accuracy trade-offs.

## 📋 Overview

MDP introduces two key innovations:
- **Multi-Granularity Pruning (MGP)**: Considers both channel-level and block-level pruning for more flexible model compression
- **Multi-Dimensional Latency Modeling (MDLM)**: Uses advanced nonlinear latency modeling for more accurate latency prediction

## 🚀 Setup and Requirements

### 1. Environment Setup

**Option A: Docker**
Build the Docker image using the provided [Dockerfile](Dockerfile)

**Option B: Virtual Environment**

Create a virtual environment with Python 3.6 and install the required packages:

```bash
# Core deep learning dependencies
pip install torch==1.4.0
pip install torchvision==0.5.0
pip install numpy
pip install Pillow
pip install PyYAML
pip install pandas
```

**Mixed Precision Support**
Install NVIDIA APEX for FP16 support: [NVIDIA APEX Installation Guide](https://github.com/NVIDIA/apex#quick-start)

**Optimization Solver Dependencies**
For solving Mixed-Integer Nonlinear Programming (MINLP) problems, we use Pyomo with MindtPy framework:

```bash
# Install optimization packages
pip install pulp
pip install pyomo

# Install solvers (Linux)
apt install -y gcc libglpk-dev glpk-utils
conda install -c conda-forge ipopt
```

> 💡 **Verification**: Test your MINLP setup with the [Pyomo MindtPy example](https://pyomo.readthedocs.io/en/6.8.0/contributed_packages/mindtpy.html)

### 2. Download Required Assets

**Pretrained Models**
Download the baseline ResNet50 model from [Google Drive](https://drive.google.com/file/d/1umlp0kUcRDBy1fABKHyU5zn2j3CvYb3L/view?usp=drive_link) and note the local path.

**Latency Lookup Tables (LUTs)**
Download the latency LUTs from [Google Drive](https://drive.google.com/file/d/1kd5MhxshRz6HCdizCLzTCAQoRrEX2W5v/view?usp=drive_link) and note the local path.

**ImageNet Dataset**
Download [ImageNet1K](https://image-net.org/download.php) and extract it to your desired location.



## 🏃‍♂️ Running Experiments

> **Important**: Update the file paths in the commands below to match your local setup:
> - `--data-root`: Path to your ImageNet dataset
> - `--pre-trained`: Path to your downloaded pretrained model
> - `--lut-file`: Path to your downloaded latency LUT file
> - `--output-dir`: Your desired output directory

### 🎯 Full MDP Method (Recommended)

This runs our complete **Multi-Dimensional Pruning** approach, which combines both innovations for optimal latency-accuracy trade-offs. This is the proposed approach in the paper.

**Key Features:**
- Multi-Granularity Pruning (MGP): Channel + block-level pruning
- Multi-Dimensional Latency Modeling (MDLM): Advanced nonlinear latency modeling

**Related Files:**
- `training_mdp.py` - Training script for MDP
- `latency_targeted_pruning_mdp.py` - MDP pruning implementation

**Command:**
```bash
python multiproc.py \
    --exp-name mdp_70 \
    --output-dir /path/to/results \
    main.py \
    --arch resnet50 \
    --amp \
    --data-root /path/to/ImageNet2012/ \
    --prune-ratio 0.7 \
    --pre-trained /path/to/imagenet_resnet50_full_model.pth \
    --lut-file /path/to/cudnn_v7.4_conv_LUT_repeat_100_step_2_scope_2048_batch256_forward_resnet50_ngc.pkl \
    --mdp
```

### 🔬 Ablation Studies

Our ablation studies systematically evaluate the contribution of each component in our MDP framework through four configurations:

| Configuration | Pruning Method | Latency Modeling | Key Features | Status |
|---------------|----------------|------------------|--------------|--------|
| **Baseline (HALP)** | OCP | LLM | Traditional latency-aware channel-only pruning: HALP | ⚪ Baseline |
| **MGP + LLM** | MGP | LLM | Our multi-granularity pruning only | 🟡 Partial |
| **OCP + MDLM** | OCP | MDLM | Our advanced latency modeling only | 🟡 Partial |
| **Full MDP** | MGP | MDLM | Complete framework with both innovations | 🟢 **Recommended** |

For detail, please refer to Section 4.5 Ablation Study in the paper.

#### 📝 Method Abbreviations
- **MGP**: Multi-Granularity Pruning *(enables block-level pruning)*
- **MDLM**: Multi-Dimensional Latency Modeling *(nonlinear latency prediction)*  
- **OCP**: Only Channel Pruning *(traditional approach)*
- **LLM**: Linear Latency Modeling *(baseline approach)*

> **Technical Note**: MINLP problems are solved using the Pyomo MindtPy framework with GLPK for linear subproblems and IPOPT for nonlinear subproblems.

#### Configuration Commands

**1. Baseline (HALP) - OCP + LLM**
```bash
python multiproc.py \
    --exp-name halp70 \
    --output-dir /path/to/results \
    main.py \
    --arch resnet50 \
    --amp \
    --data-root /path/to/ImageNet2012/ \
    --prune-ratio 0.7 \
    --pre-trained /path/to/imagenet_resnet50_full_model.pth \
    --lut-file /path/to/cudnn_v7.4_conv_LUT_repeat_100_step_2_scope_2048_batch256_forward_resnet50_ngc.pkl
```

**2. MGP + LLM Configuration**

*Related Files:*
- `training_mgp.py` - Training script with multi-granularity pruning
- `latency_targeted_pruning_mgp.py` - MGP implementation

```bash
python multiproc.py \
    --exp-name mgp_70 \
    --output-dir /path/to/results \
    main.py \
    --arch resnet50 \
    --amp \
    --data-root /path/to/ImageNet2012/ \
    --prune-ratio 0.7 \
    --pre-trained /path/to/imagenet_resnet50_full_model.pth \
    --lut-file /path/to/cudnn_v7.4_conv_LUT_repeat_100_step_2_scope_2048_batch256_forward_resnet50_ngc.pkl \
    --mgp
```

**3. OCP + MDLM Configuration**

*Related Files:*
- `training_mdlm.py` - Training script with advanced latency modeling
- `latency_targeted_pruning_mdlm.py` - MDLM implementation

```bash
python multiproc.py \
    --exp-name mdlm_70 \
    --output-dir /path/to/results \
    main.py \
    --arch resnet50 \
    --amp \
    --data-root /path/to/ImageNet2012/ \
    --prune-ratio 0.7 \
    --pre-trained /path/to/imagenet_resnet50_full_model.pth \
    --lut-file /path/to/cudnn_v7.4_conv_LUT_repeat_100_step_2_scope_2048_batch256_forward_resnet50_ngc.pkl \
    --mdlm
```

## 📊 Latency Measurement

After pruning, measure the actual inference latency using the generated mask file. The mask file will be saved in your specified `--output-dir`.

### For Block Pruning Methods
Use this for configurations that include **MGP** (multi-granularity pruning):
- ✅ **Full MDP** (MGP + MDLM)
- ✅ **MGP + LLM** configuration

```bash
python3 est_inference.py \
    -a resnet50 \
    --batch_size 256 \
    --mask /path/to/your/mask_file.pkl \
    --mgp
```

### For Channel-Only Pruning Methods
Use this for configurations with **OCP** (only channel pruning):
- ✅ **Baseline (HALP)** (OCP + LLM)
- ✅ **OCP + MDLM** configuration

```bash
python3 est_inference.py \
    -a resnet50 \
    --batch_size 256 \
    --mask /path/to/your/mask_file.pkl
```

> 💡 **Tip**: The mask file location will be printed during the pruning process, or you can find it in your specified output directory.