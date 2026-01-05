#!/bin/bash
#SBATCH -p $vp
#SBATCH --gres=gpu:4

srun -p $vp --gres=gpu:4 apptainer exec --nv ~/ubuntu.sif bash -c "
source ~/.bashrc && \
conda activate poze && \
cd /mnt/petrelfs/zhangsiyu/4dgen/poze && \
python visualize_ply_sequence.py \
    --input ./outputs/snowball_fc_2.ply \
    --save ./outputs/snowball_preview.mp4 \
    --max-points 60000 \
    --fps 6 \
    --point-size 0.6 \
    --pad-fraction 0.05
"
