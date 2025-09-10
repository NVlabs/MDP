# tools/dist_train.sh projects/configs/StreamPETR/stream_petr_r50_flash_704_bs2_seq_428q_nui_24e.py 8 --work-dir /result/prune_test/ --resume-from /workspace/alex/StreamPETR/test/reported_stream_petr_r50_flash_704_bs2_seq_428q_nui_60e.pth
tools/dist_train.sh projects/configs/StreamPETR/stream_petr_r50_flash_704_bs2_seq_428q_nui_60e_prune.py 8 --work-dir /result/halppp_prune_70/
