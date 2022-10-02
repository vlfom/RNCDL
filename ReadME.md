# Learning to Discover and Detect Objects

This repository contains an official implementation of the Region-based Novel Class Discovery, Detection and Localization (RNCDL) method.

Our implementation is based on the [Detectron2](https://github.com/facebookresearch/detectron2) framework.

# Installation

Please follow [Detectron2's setup guide](https://github.com/facebookresearch/detectron2/blob/main/INSTALL.md) and pick the **CUDA=11.3** and **torch=1.10** versions (e.g. `python -m pip install detectron2 -f \
  https://dl.fbaipublicfiles.com/detectron2/wheels/cu113/torch1.10/index.html` may work). We do not require any additional dependencies, apart from the default ones used by Detectron2.

## Download datasets

For COCO + LVIS experiments, download COCO and LVIS datasets as instructed in the [Detectron2 data documentation](https://detectron2.readthedocs.io/en/latest/tutorials/builtin_datasets.html).

**Note on LVIS + VisualGenome experiments:** LVIS and VisualGenome datasets largely overlap (50K images). VisualGenome does not provide a default train-val split and for our setup we devised a specific split so that the validation images are those that appear in LVIS validation split and can be found in VisualGenome. For more details, please see our paper (supplementary).

For LVIS + VisualGenome experiments, first download LVIS dataset as instructed above. For VisualGenome dataset we had to pre-process it to match the Detectron2 format and thus it's required to download our custom annotation files in addition to the images provided in the original dataset. First, download the VisualGenome images from the [official website](#). Then, put them **in the same folder where LVIS images are located**. Then, download our prepared annotations from [this link](#). Put them to the `$DETECTRON2_DATASETS$/visualgenome/` folder (if you want to use a different folder, please modify `configs/data/register_vglvis.py` file accordingly).

## Download checkpoints

We start our R-CNN training from a backbone pretrained in a self-supervised manner, specifically, MoCo v2. For that, please download 800-epoch weights from their official repository using [this link](https://dl.fbaipublicfiles.com/moco/moco_checkpoints/moco_v2_800ep/moco_v2_800ep_pretrain.pth.tar) (obtained from [this table](https://github.com/facebookresearch/moco#models)), clone the [official moco repository](https://github.com/facebookresearch/moco), and [run a script as described here](https://github.com/facebookresearch/moco/tree/main/detection) to convert the weights to the Detectron2 format.

# Running scripts

We use [Detectron2's lazy configuration](https://detectron2.readthedocs.io/en/latest/tutorials/lazyconfigs.html) to define our framework. The configurations are located in `configs/` folder and the root config files can be found in `config/train/`.

For convenience, we provide [slurm](https://slurm.schedmd.com/documentation.html) example job scripts that are based on the scripts that we used to execute the jobs. They can be found in `slurm_scripts/`.
All our experiments were tested on 4 NVIDIA A40 GPUs with 48G memory.

To train our fully-supervised baselines, please use scripts in the `slurm_scripts/fully_supervised/` folder. E.g. to train a fully-supervised R-CNN on COCO<sub>half</sub> use:

```
python tools/train_supervised.py \
    \
    --config-file ./configs/train/fully_supervised/coco50pct.py \
    --num-gpus 4 \
    --dist-url 'tcp://localhost:10042' \
    \
    train.init_checkpoint=./checkpoints/moco_v2_800ep_pretrain.pkl \
    train.output_dir=./output \
    train.exp_name="coco50pct-fully_super" \
    train.eval_period=20000
```

To train our discovery networks, please use scripts in the `slurm_scripts/discovery/` folder. E.g. to run discovery training for COCO<sub>half</sub> + LVIS setup use:

```
python tools/train_discovery.py \
    \
    --config-file ./configs/train/discovery/coco50pct_lvis.py \
    --num-gpus 4 \
    --dist-url 'tcp://localhost:10042' \
    \
    train.exp_name="coco50pct_lvis-discovery" \
    train.output_dir=./output \
    train.eval_period=999999 \
    discovery_evaluator.evaluator.output_dir=./output \
    \
    train.init_checkpoint=./checkpoints/coco50pct_pretrain.pkl \
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

```
