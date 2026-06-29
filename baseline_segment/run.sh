#$ -S /bin/bash
#$ -j y
#$ -N baseline_unet
#$ -l tmem=16G
#$ -l gpu=true
#$ -l h_rt=20:00:00
#$ -wd /cluster/project7/longitude/atlasSam/baseline_segment
#$ -o /cluster/project7/longitude/atlasSam/baseline_segment/logs/$JOB_NAME.$JOB_ID.log

hostname
nvidia-smi

echo "Selected GPU $GPU_ID"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate /cluster/project7/longitude/env

export PYTHONUNBUFFERED=1
mkdir -p logs checkpoints

python -u baseline_unet.py \
  --phase train \
  --data_root ../../Datasets/ProstateDatasets/data_split_files \
  --path 1-uclH-data_ratio0.8 \
  --gpus "${GPU_ID:-0}" \
  --epochs 100 \
  --batch_size 4 \
  --spacing 0.5 0.5 1.0 \
  --base_channels 24 \
  --lr 1e-4 \
  --save_freq 10
