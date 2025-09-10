# LookAhead for Object Detection

## Code
Logic written in optimizer.py. The latency inference code in this repo is not updated with the latest one. Sent the latest inference code to Maying before, so pls check with her for the model cleaning and infer code.
## How to Run
```
python train.py --data-dir /raid/json_path/ --prune-start [PRUNE_START] --prune-end [PRUNE_END] --method 26 --reg-conf configs_coarse/resnet50_backbone.json --arch resnet50 --coarse-pruning --epochs [TOTAL_EPOCHS] --model ssd512 --batch-size 128 -m 26 --prune-ratio [PRUNE_RATIO] --backbone-only --output-dir /result
```

Example:
```
python train.py --data-dir /raid/json_path/ --prune-start 100 --prune-end 200 --method 26 --reg-conf configs_coarse/resnet50_backbone.json --arch resnet50 --coarse-pruning --epochs 900 --model ssd512 --batch-size 128 -m 26 --prune-ratio 0.79 --backbone-only --output-dir /result
```

## Eval
```
python eval.py --data-dir /raid -a resnet50 --print-stats -s [CKPT_START] -e [CKPT_END] --checkpoint-dir [CKPT_DIR]
```
Example
```
python eval.py --data-dir /raid -a resnet50 --print-stats -s 870 -e 899 --checkpoint-dir /result/2022-11-09_14-12-09
```
