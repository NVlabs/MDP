# python train.py --data-dir /raid/json_path/ --prune-start 100 --prune-end 200 --method 26 --reg-conf configs_coarse/resnet50_backbone.json --arch resnet50 --coarse-pruning --epochs 900 --model ssd512 --batch-size 128 -m 26 --prune-ratio 0.8 --backbone-only --output-dir /result

# python train.py --data-dir /raid/json_path/ --prune-start 150 --prune-end 250 --method 26 --reg-conf configs_coarse/resnet50_backbone.json --arch resnet50 --coarse-pruning --epochs 900 --model ssd512 --batch-size 128 -m 26 --prune-ratio 0.79 --backbone-only --output-dir /result

python train.py --data-dir /raid/json_path/ --prune-start 100 --prune-end 200 --method 26 --reg-conf configs_coarse/resnet50_backbone.json --arch resnet50 --coarse-pruning --epochs 900 --model ssd512 --batch-size 128 -m 26 --prune-ratio 0.79 --backbone-only --output-dir /result
