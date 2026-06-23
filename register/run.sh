#$ -S /bin/bash
#$ -j y
#$ -N register
#$ -l tmem=16G
#$ -l gpu=true
#$ -l gpu_type=!(v100|titanxp|titanx|l40s)
#$ -l h_rt=50:00:00
#$ -wd /cluster/project7/longitude/atlasSam/register
#$ -o /cluster/project7/longitude/atlasSam/register/logs/$JOB_NAME.$JOB_ID.log

hostname
nvidia-smi

echo "Selected GPU $GPU_ID"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate ../env

export PYTHONUNBUFFERED=1
mkdir -p logs checkpoints atlas

python -u register_atlas.py \
  --phase train \
  --gpus "${GPU_ID:-0}" \
  --epochs 50 \
  --batch_size 4 \
  --save_freq 1 \
  --atlas_update_freq 5 \
  --label_loss_weight 0.1
