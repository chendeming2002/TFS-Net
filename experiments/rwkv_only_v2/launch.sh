#!/bin/bash
# Launch RWKV-Only v2 training
cd /home/a1005/25/TFS-Net
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
nohup /home/a1005/anaconda3/envs/ptorch/bin/python -u experiments/rwkv_only_v2/train.py --config experiments/rwkv_only_v2/config.yaml &>/tmp/rwkv_v2.log &
echo "Launched PID=$!"
