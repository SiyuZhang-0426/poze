#!/bin/bash
#SBATCH -p $vp
#SBATCH --gres=gpu:4

srun -p $vp --gres=gpu:4 apptainer exec --nv ~/ubuntu.sif bash -c "
source ~/.bashrc && \
conda activate poze && \
cd /mnt/petrelfs/zhangsiyu/4dgen/poze && \
python finetune.py \
    --wan-ckpt-dir ./Wan2.2-TI2V-5B \
    --dataset-root ./data/wan21 \
    --dataset-use-latents true \
    --batch-size 1 \
    --steps 200 \
    --latent-weight 1.0 \
    --video-weight 0.5
"
