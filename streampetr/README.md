# MINLP-Block-Prune-Nuscenes

This repository is for the improved HALP formulation with:

    1. Consideration of input channel reduction in pruning
    2. Block Pruning
    3. Oneshot Pruning
    4. Soft Masking

Specifically, these are enabled by formulating pruning as a Mixed Integer Nonlinear Program (MINLP). For solving MINLP, we leverage Pyomo MindtPy framework for decomposition then glpk for linear parts and ipopt for nonlinear parts.

This repo contains the Nuscenes experiments, specifically StreamPetr. 

Contact: Alex Sun (xinglongs@nvidia.com), Maying Shen (mshen@nvidia.com), Shiyi Lan (shiyil@nvidia.com), Joshua Chen (joshchen@nvidia.com)

Please follow the below steps when running this codebase.


## Step 0. Mount My NGC Workspace
```
ngc workspace mount alex-space alex-space/ --mode RW
```

Specifically, you would need the ```*temporal*.pkl``` in my NGC repo.

## Step 1. Request NGC Resources

Run:
```
bash alex-imgnet-nuscenes.sh
```
with your NGC workspace and credentials.

## Step 2. Set up the environments
In NGC Jupyter, run:
```
bash pkgs_streampetr.sh
```
This installs pyomo and other packages. Also, this copies things like pretrained models and lookup tables from the aforementioned saved_models folder to /home.
Please carefully read through the file first to see if you need to change any name, for example, I used the address ```/workspace/alex/StreamPETR```. Maybe you want another address.

The above file creates a ```streampetr``` environment. You need to activate it before running your code:
```
source ~/.bashrc
conda activate streampetr
```

## Step 3. Run Experiments

### Notes
I implemented pruning for mmcv and mmdetection3 using hook.

Specifically, check ```projects/mmdet3d_plugin/core/apis/mmdet_train.py```, ```Lines 208-217```, about how I link the hook to the entire training workflow.

### HALP Baseline Example
Related files: 

    1. projects/configs/StreamPETR/stream_petr_r50_flash_704_bs2_seq_428q_nui_60e_prune_halp.py 
    2. prune/prune_halp.py

Run the command:
```
tools/dist_train.sh projects/configs/StreamPETR/stream_petr_r50_flash_704_bs2_seq_428q_nui_60e_prune_halp.py 8 --work-dir /result/halp_prune_70/
```
Change ```self.prune_ratio``` in prune/prune_halp.py and other defined handles for different pruning hyperparams.

### Run HALP2 + Block Pruning.
Related files: 

    1. projects/configs/StreamPETR/stream_petr_r50_flash_704_bs2_seq_428q_nui_60e_prune.py 
    2. prune/prune.py

This is the combined version (block pruning under MINLP) with the best performance (considering no soft masking involved). 

```
tools/dist_train.sh projects/configs/StreamPETR/stream_petr_r50_flash_704_bs2_seq_428q_nui_60e_prune.py 8 --work-dir /result/halp2_blkprune_70/
```
Change ```self.prune_ratio``` in prune/prune_halp.py and other defined handles for different pruning hyperparams.

## Measure Latency?
Please follow the instructions from another repo(https://gitlab-master.nvidia.com/xinglongs/measure-streampetr-fps) on measuring pruned StreamPetr latency.