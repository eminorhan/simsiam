#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=240GB
#SBATCH --time=00:10:00
#SBATCH --job-name=train_simsiam
#SBATCH --output=train_simsiam_%A_%a.out
#SBATCH --array=0

export MASTER_ADDR=$(hostname -s)
export MASTER_PORT=$(shuf -i 10000-65500 -n 1)
export WORLD_SIZE=1

DATAS=(
	"{00000..99999}" 
	)

SAVES=(
	"simsiam_vimlp_liaon2b" 
	)

DATA=${DATAS[$SLURM_ARRAY_TASK_ID]}
SAVE=${SAVES[$SLURM_ARRAY_TASK_ID]}

echo $DATA
echo $SAVE

srun python -u ../train.py \
	--resume "" \
	--batch_size_per_gpu 512 \
	--num_workers 16 \
	--lr 0.0001 \
	--min_lr 0.0001 \
	--weight_decay 0.0 \
	--output_dir ../outputs \
	--data_path "/scratch/work/public/ml-datasets/laion2B-en-data/${DATA}.tar" \
	--save_prefix "${SAVE}"

echo "Done"