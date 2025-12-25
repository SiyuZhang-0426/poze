#!/bin/bash
#SBATCH -p $vp
#SBATCH --gres=gpu:4

srun -p $vp --gres=gpu:4 apptainer exec --nv ~/ubuntu.sif bash -c "
source ~/.bashrc && \
conda activate poze && \
cd /mnt/petrelfs/zhangsiyu/4dgen/poze && \
python ./pi3/example.py \
    --ckpt ./pi3_ckpt/model.safetensors \
    --data_path ./zsydata
"
