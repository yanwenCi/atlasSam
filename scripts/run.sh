#$ -S /bin/bash
#$ -j y
#$ -N atlas_bank
#$ -l tmem=8G
#$ -l gpu=true
#$ -l h_rt=1:00:00
#$ -wd /cluster/project7/longitude/atlasSam/scripts
#$ -o /cluster/project7/longitude/atlasSam/scripts/logs/$JOB_NAME.$JOB_ID.log
#$ -e /cluster/project7/longitude/atlasSam/scripts/logs/$JOB_NAME.$JOB_ID.log

hostname

source ~/miniconda3/etc/profile.d/conda.sh
conda activate ../../env

export PYTHONUNBUFFERED=1
mkdir -p logs ../register/atlas_bank

python -u generate_atlas_bank.py \
  --case_list /cluster/project7/longitude/Datasets/ProstateDatasets/1-uclH-data/data-ROI-192-96/data_split_files/data_include_missing/path_list.txt \
  --data_root /cluster/project7/longitude/Datasets/ProstateDatasets/1-uclH-data/data-ROI-192-96 \
  --output_dir ../register/atlas_bank \
  --n_clusters 6 \
  --n_per_cluster 5 \
  --pz_label 1 \
  --cg_label 2


# python -u generate_atlas_bank.py \
#   --case_list /raid/candi/Wen/Dataset/ProstateDatasets/3-picai-data/data-ROI-192-96/data_split_files/data_include_missing/path_list.txt \
#   --data_root /raid/candi/Wen/Dataset/ProstateDatasets/3-picai-data/data-ROI-192-96 \
#   --output_dir ../register/atlas_bank \
#   --n_clusters 6 \
#   --n_per_cluster 3 \
#   --pz_label 2 \
#   --cg_label 1
