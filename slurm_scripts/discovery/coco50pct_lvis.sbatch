#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:4,VRAM:48G
#SBATCH --mem=200GB

module load cuda/11.3.1

source /usr/miniconda3/etc/profile.d/conda.sh
conda activate detectron2

export DETECTRON2_DATASETS=/storage/slurm/data

OUTPUT_DIR_PREF=/storage/slurm/slurm_runs/coco50pct_lvis-discover
OUTPUT_DIR_SUFF=$(tr -dc A-Za-z0-9 </dev/urandom | head -c 5 ; echo '')
OUTPUT_DIR="$OUTPUT_DIR_PREF-$OUTPUT_DIR_SUFF"

CONFIG_FILE="./configs/train/discovery/coco50pct_lvis.py"
CHECKPOINT_PATH="/usr/detection/coco50pct_pretrain.pkl"

python tools/train_discovery.py \
    \
    --config-file $CONFIG_FILE \
    --num-gpus 4 \
    --dist-url 'tcp://localhost:10042' \
    \
    train.exp_name="coco50pct_lvis-discovery" \
    train.output_dir=$OUTPUT_DIR \
    train.eval_period=999999 \
    discovery_evaluator.evaluator.output_dir=$OUTPUT_DIR \
    \
    train.init_checkpoint=$CHECKPOINT_PATH \
    train.max_iter=15000 \
    optimizer.lr=0.01 \
    train.seed=42 \
    \
    model_proposals_extraction_param.test_nms_thresh=1.01 \
    model_proposals_extraction_param.test_topk_per_image=50 \
    \
    train.supervised_loss_lambda=0.5 \
    \
    model_supervised.roi_heads.box_predictor.discovery_model.num_unlabeled=3000 \
    model_supervised.roi_heads.box_predictor.discovery_model.sk_mode="lognormal" \
    \
    model_supervised.roi_heads.box_predictor.discovery_model.memory_batches=100 \
    model_supervised.roi_heads.box_predictor.discovery_model.memory_patience=150
