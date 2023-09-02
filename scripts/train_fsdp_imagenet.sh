#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:a100:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=480GB
#SBATCH --time=2:00:00
#SBATCH --job-name=train_simsiam_fsdp_imagenet
#SBATCH --output=train_simsiam_fsdp_imagenet_%A_%a.out
#SBATCH --array=0

export MASTER_ADDR=$(hostname -s)
export MASTER_PORT=$(shuf -i 10000-65500 -n 1)
export WORLD_SIZE=4

SAVES=(
	"simsiam_vimlp_imagenet" 
	)

SAVE=${SAVES[$SLURM_ARRAY_TASK_ID]}

echo $SAVE

srun python -u ../train_fsdp.py \
	--resume "" \
	--batch_size_per_gpu 512 \
	--num_workers 16 \
	--lr 0.0001 \
	--weight_decay 0.0 \
	--dim 2048 \
	--pred_dim 512 \
	--output_dir ../outputs \
	--data_path "/scratch/eo41/data/imagenet/imagenet_train_{000000..000001}.tar" \
	--save_prefix "${SAVE}"

echo "Done"