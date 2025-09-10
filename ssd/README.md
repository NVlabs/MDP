# MINLP-Block-Prune-SSD

This codebae is MINLP-Block-Prune instantiated on Pascal 2D detection and SSD network.

This repository is for the improved HALP formulation with:

    1. Consideration of input channel reduction in pruning
    2. Block Pruning
    3. Oneshot Pruning
    4. Soft Masking

Specifically, these are enabled by formulating pruning as a Mixed Integer Nonlinear Program (MINLP). For solving MINLP, we leverage Pyomo MindtPy framework for decomposition then glpk for linear parts and ipopt for nonlinear parts.

This repo contains the Pascal VOC experiments, specifically SSD512-RN50. 

Contact: Alex Sun (xinglongs@nvidia.com), Maying Shen (mshen@nvidia.com), Joshua Chen (
joshchen@nvidia.com)

Please follow the below steps when running this codebase.

## Code
Logic written in optimizer.py. 

## Step 0. Mount My NGC Workspace
```
ngc workspace mount alex-space alex-space/ --mode RW
```

Specifically, you would need the saved_models folder in my NGC repo.

## Step 1. Request NGC Resources

Run:
```
bash alex-pascal.sh
```
OR
```
bash alex-pascal2.sh  # small 16G GPU
```
with your NGC workspace and credentials.

## Step 2. Set up the environments
In NGC Jupyter, run:
```
bash pkgs_pascal.sh
```
This installs pyomo and other packages. Also, this copies things like pretrained models and lookup tables from the aforementioned saved_models folder to /home.

## Step 2. Run Experiments


### Run HALP2 + Block Pruning.
Related files: 

    1. optimizer.py 
    2. latency_targeted_pruning.py

This is the combined version (block pruning under MINLP) with the best performance (considering no soft masking involved). Default version is with oneshot pruning. 

```
python train.py --data-dir /raid/json_path/ --prune-start 0 --prune-end 200 --method 26 --reg-conf configs_coarse/resnet50_backbone.json --arch resnet50 --coarse-pruning --epochs 900 --model ssd512 --batch-size 128 -m 26 --prune-ratio 0.7 --backbone-only --output-dir /result --load-ckpt /home/dense_ssd512.pth
```

Adjust the pruning ratio accordingly.

## Measure Latency?
You need the mask file after pruning.

For measuring a layer_masks.pkl latency, you need:
```
python3 est_inference_pascal_hahp.py
```
I was lazy during this code and didn't set up command line argument. Change the line 62 in est_inference_pascal_hahp.py to point to the layer mask you want to measure. 

Also, importantly, if you see error complaining the tensor shape does not match, you would need to change line 394 in ```ssd_models/models.py```. The argument out_channels defines the number of output channels of the backbone, you can change this value according to the error reporting. This is a bit confusing, check with Maying on clarifications on this. Main reason is we also prune the final output channels of the ResNet backbone.
