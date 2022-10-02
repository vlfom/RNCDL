#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:4,VRAM:48G
#SBATCH --mem=200GB

module load cuda/11.3.1

source /usr/miniconda3/etc/profile.d/conda.sh
conda activate detectron2

export DETECTRON2_DATASETS=/storage/slurm/data

CONFIG_FILE=./configs/train/fully_supervised/coco50pct.py

OUTPUT_DIR_PREF=/storage/slurm/slurm_runs/coco50pct-fully_super
OUTPUT_DIR_SUFF=$(tr -dc A-Za-z0-9 </dev/urandom | head -c 5 ; echo '')
OUTPUT_DIR="$OUTPUT_DIR_PREF-$OUTPUT_DIR_SUFF-$SLURM_JOB_ID"

CHECKPOINT_PATH="/usr/detection/moco_v2_800ep_pretrain.pkl"

python tools/train_supervised.py \
    --config-file $CONFIG_FILE \
    --num-gpus 4 \
    --dist-url 'tcp://localhost:10042' \
    train.init_checkpoint=$CHECKPOINT_PATH \
    train.output_dir=$OUTPUT_DIR \
    train.exp_name="coco50pct-fully_super" \
    train.eval_period=20000
