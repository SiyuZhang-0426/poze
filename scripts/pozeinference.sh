#!/bin/bash
#SBATCH -p $vp
#SBATCH --gres=gpu:4

srun -p $vp --gres=gpu:4 apptainer exec --nv ~/ubuntu.sif bash -c "
source ~/.bashrc && \
conda activate poze && \
cd /mnt/petrelfs/zhangsiyu/4dgen/poze && \
python inference.py \
    --wan-ckpt-dir ./Wan2.2-TI2V-5B \
    --pi3-checkpoint ./pi3_ckpt/model.safetensors \
    --frame-num 24 \
    --image ./zsydata/snowball.jpg \
    --prompt 'Ultra-detailed close-up of a fluffy white Persian kitten with long, silky fur and large dark eyes, lying on a glossy white surface. The kitten gently wags its tail and shifts its gaze curiously. Background: modern minimalist living room with a warm spherical lamp, blurred beige furniture, and soft natural lighting. 4K resolution, cinematic shallow depth of field, smooth motion, cozy serene atmosphere, ultra-realistic fur texture.' \
    --device cuda \
    --use-pi3 True \
    --concat-method width \
    --output ./outputs/snowball.mp4 \
    --save-pi3 ./outputs/snowball.ply
"
