# python -m torch.distributed.launch --nproc_per_node=4 --use_env main_full_global_latency.py --model deit_base_distilled_patch16_224 --epochs 50 --num_workers 10 --batch-size 128 --data-path /raid/ImageNet2012/ImageNet2012 --data-set IMNET --lr 1e-4 --output_dir /result/ --amp --input-size 224 --seed 1 --pruning_config=pruning_configs/group8_m23_m09.json --prune_per_iter=32 --kl_loss_coeff=100000 --original_loss_coeff=1.0 --student_eval=True --dist-eval --pruning --prune_dict '{"Global":39000}' --interval_prune 100 --pretrained --distillation-type hard --latency_regularization 5e-4 --latency_target 0.50 --latency_look_up_table latency_head.json --pruning_exit

CUDA_VISIBLE_DEVICES=0 python eval_nvit.py --finetune /workspace/alex/mdp_transformer_results/global_prune_IMNET_deit_base_distilled_patch16_224_lr1e-05_FLOPs_80x200_lr1e-5_Global_39000_lat_0.0005_target_0.07_ft_50/ft_dense_100_lr_0.0002_a_0.5_T_20.0/ft_checkpoint.pth --data-path /raid/ImageNet2012/ImageNet2012 --batch-size 256

CUDA_VISIBLE_DEVICES=0 python eval_nvit.py --finetune /workspace/alex/mdp_transformer_results/global_prune_IMNET_deit_base_distilled_patch16_224_lr1e-05_FLOPs_80x200_lr1e-5_Global_39000_lat_0.0005_target_0.23_ft_50/ft_dense_100_lr_0.0002_a_0.5_T_20.0/ft_checkpoint.pth --data-path /raid/ImageNet2012/ImageNet2012 --batch-size 256


CUDA_VISIBLE_DEVICES=0 python eval_nvit.py --finetune /workspace/alex/mdp_transformer_results/global_prune_IMNET_deit_base_distilled_patch16_224_lr0.0001_50x200_Global_39000_lat_0.0005_target_0.19_ft_50/ft_dense_300_lr_0.0002_a_0.5_T_20.0/ft_checkpoint.pth --data-path /raid/ImageNet2012/ImageNet2012 --batch-size 256

CUDA_VISIBLE_DEVICES=0 python eval_nvit.py --finetune /workspace/alex/mdp_transformer_results/global_prune_IMNET_deit_base_distilled_patch16_224_lr1e-05_80x200_lr1e-5_Global_39000_lat_0.0005_target_0.19_ft_50/ft_dense_300_lr_0.0002_a_0.5_T_20.0/ft_checkpoint.pth --data-path /raid/ImageNet2012/ImageNet2012 --batch-size 256

CUDA_VISIBLE_DEVICES=0 python eval_nvit.py --finetune /workspace/alex/mdp_transformer_results/global_prune_IMNET_deit_base_distilled_patch16_224_lr1e-05_80x200_lr1e-5_Global_39000_lat_0.0005_target_0.39_ft_50/ft_dense_300_lr_0.0002_a_0.5_T_20.0/ft_checkpoint.pth --data-path /raid/ImageNet2012/ImageNet2012 --batch-size 256


CUDA_VISIBLE_DEVICES=0 python eval_nvit.py --finetune /workspace/alex/global_prune_IMNET_deit_base_distilled_patch16_224_lr0.0001_Global_39000_lat_0.0005_target_0.39_ft_50/ft_dense_100_lr_0.0002_a_0.5_T_20.0/ft_checkpoint.pth --data-path /raid/ImageNet2012/ImageNet2012 --batch-size 256

CUDA_VISIBLE_DEVICES=0 python eval_nvit.py --finetune /workspace/alex/global_prune_IMNET_deit_base_distilled_patch16_224_lr0.0001_Global_39000_lat_0.0005_target_0.19_ft_50/ft_dense_100_lr_0.0002_a_0.5_T_20.0/ft_checkpoint.pth --data-path /raid/ImageNet2012/ImageNet2012 --batch-size 256

# Ours
CUDA_VISIBLE_DEVICES=0 python eval_nvit.py --finetune /result/global_prune_IMNET_deit_base_distilled_patch16_224_lr0.0001_50x200_Global_39000_lat_0.0005_target_0.19_ft_50/ft_dense_100_lr_0.0002_a_0.5_T_20.0/ft_checkpoint.pth --data-path /raid/ImageNet2012/ImageNet2012 --batch-size 256

CUDA_VISIBLE_DEVICES=0 python eval_nvit.py --finetune /result/global_prune_IMNET_deit_base_distilled_patch16_224_lr0.0001_50x200_Global_39000_lat_0.0005_target_0.39_ft_50/ft_dense_100_lr_0.0002_a_0.5_T_20.0/ft_checkpoint.pth --data-path /raid/ImageNet2012/ImageNet2012 --batch-size 256

# Baseline 
# CUDA_VISIBLE_DEVICES=0 python eval_nvit_baseline.py --finetune /workspace/alex/global_prune_IMNET_deit_base_distilled_patch16_224_lr0.0001_Global_39000_lat_0.0005_target_0.39_ft_50/ft_dense_100_lr_0.0002_a_0.5_T_20.0/ft_checkpoint.pth --data-path /raid/ImageNet2012/ImageNet2012 --batch-size 256

CUDA_VISIBLE_DEVICES=0 python eval_nvit_baseline_speed.py --finetune /workspace/alex/global_prune_IMNET_deit_base_distilled_patch16_224_lr0.0001_Global_39000_lat_0.0005_target_0.39_ft_50/ft_dense_100_lr_0.0002_a_0.5_T_20.0/ft_checkpoint.pth --data-path /raid/ImageNet2012/ImageNet2012 --batch-size 256

CUDA_VISIBLE_DEVICES=0 python eval_nvit_baseline_speed_trt.py --finetune /workspace/alex/global_prune_IMNET_deit_base_distilled_patch16_224_lr0.0001_Global_39000_lat_0.0005_target_0.39_ft_50/ft_dense_100_lr_0.0002_a_0.5_T_20.0/ft_checkpoint.pth --data-path /raid/ImageNet2012/ImageNet2012 --batch-size 256

CUDA_VISIBLE_DEVICES=0 python eval_nvit_baseline_speed_trt.py --finetune /home/xinglongs-ngc/alex-space/global_prune_IMNET_deit_base_distilled_patch16_224_lr0.0001_Global_39000_lat_0.0005_target_0.39_ft_50/ft_dense_100_lr_0.0002_a_0.5_T_20.0/ft_checkpoint.pth --data-path /raid/ImageNet2012/ImageNet2012 --batch-size 256

CUDA_VISIBLE_DEVICES=0 python eval_nvit_speed_trt.py --finetune /home/xinglongs-ngc/alex-space/global_prune_IMNET_deit_base_distilled_patch16_224_lr0.0001_Global_39000_lat_0.0005_target_0.39_ft_50/ft_dense_100_lr_0.0002_a_0.5_T_20.0/ft_checkpoint.pth --data-path /raid/ImageNet2012/ImageNet2012 --batch-size 256

CUDA_VISIBLE_DEVICES=0 python eval_nvit_speed_trt.py --finetune /home/xinglongs-ngc/alex-space/global_prune_IMNET_deit_base_distilled_patch16_224_lr0.0001_Global_39000_lat_0.0005_target_0.19_ft_50/ft_dense_100_lr_0.0002_a_0.5_T_20.0/ft_checkpoint.pth --data-path /raid/ImageNet2012/ImageNet2012 --batch-size 256
