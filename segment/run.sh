#$ -S /bin/bash
#$ -j y
#$ -N sammed3d_atlas_bank
#$ -l tmem=32G
#$ -l gpu=true
#$ -l gpu_type=!(v100|titanxp|titanx|l40s)
#$ -l h_rt=24:00:00
#$ -wd /cluster/project7/longitude/atlasSam/segment
#$ -o /cluster/project7/longitude/atlasSam/segment/logs/$JOB_NAME.$JOB_ID.log

hostname
nvidia-smi

echo "Selected GPU ${GPU_ID:-0}"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate ../../env

export PYTHONUNBUFFERED=1
mkdir -p logs checkpoints

python -u segment.py \
  --model_type sammed3d_atlas \
  --phase test \
  --data_root ../../Datasets/ProstateDatasets/data_split_files \
  --path 1-uclH-data_ratio0.8 \
  --gpus "${GPU_ID:-0}" \
  --atlas_bank_dir ../register/atlas_bank \
  --atlas_bank_data_root /cluster/project7/longitude/Datasets/ProstateDatasets/3-picai-data/data-ROI-192-96 \
  --atlas_bank_mode cycle \
  --pz_label 2 \
  --cg_label 1 \
  --sam_crop_size 128 \
  --atlas_prompt_pad 8 \
  --sam_prob_threshold 0.5 \
  --save_dir checkpoints \
  --load_from_dir sammed3d_atlas_picai_zones_ratio0.7 \
  --save_output
