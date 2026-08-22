#!/bin/bash
cd /home/a1005/25/TFS-Net
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
exec /home/a1005/anaconda3/envs/ptorch/bin/python -u experiments/rwkv_only_v3/train.py --config experiments/rwkv_only_v3/config.yaml
