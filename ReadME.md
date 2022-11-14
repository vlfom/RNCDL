# Learning to Discover and Detect Objects

<img src="./docs/assets/figures/qualitative_results.webp" class="qualitative_teaser">

This repository provides the official implementation of the following paper:

**Learning to Discover and Detect Objects** <br/>
[Vladimir Fomenko](https://github.com/vlfom/),
[Ismail Elezi](https://dvl.in.tum.de/team/elezi/),
[Deva Ramanan](https://www.cs.cmu.edu/~deva/),
[Laura Leal-Taixé](https://dvl.in.tum.de/team/lealtaixe/),
[Aljoša Ošep](https://dvl.in.tum.de/team/osep/) <br/>
In Advances in Neural Information Processing Systems 36 (NeurIPS 2022). <br/>
[Project page](https://vlfom.github.io/RNCDL/) | [Paper](https://arxiv.org/abs/2210.10774) | [Source code](https://github.com/vlfom/rncdl) | [Poster](https://drive.google.com/file/d/1gDwgv-nKM3btpCqPU98RdC7ztuHu2cYF/view) | [Video](https://www.youtube.com/watch?v=zWpUnXNplfQ) <br/>

> **Abstract**: We tackle the problem of novel class discovery, detection, and localization (NCDL). In this setting, we assume a source dataset with labels for objects of commonly observed classes. Instances of other classes need to be discovered, classified, and localized automatically based on visual similarity, without human supervision. To this end, we propose a two-stage object detection network RNCDL, that uses a region proposal network to localize potential objects and classify them. We train our network to classify each proposal, either as one of the known classes, seen in the source dataset, or one of the extended set of novel classes with a constraint that the distribution of class assignments should follow natural long-tail distributions common in the real open-world. By training our detection network with this objective in an end-to-end manner, it learns to classify all region proposals for a large variety of classes, including those that are not part of the labeled object class vocabulary.

# Installation

Our implementation is based on the [Detectron2 v0.6](https://github.com/facebookresearch/detectron2) framework. For our experiments we followed [Detectron2's setup guide](https://github.com/facebookresearch/detectron2/blob/main/INSTALL.md) and picked the **CUDA=11.3** and **torch=1.10** versions (`python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu113/torch1.10/index.html` may work). Other Detectron2 versions should work as well, but we did not test them. All the additional dependencies we put to `requirements.txt`. We used Python 3.8 for all experiments.

## Inplace modifications of Detectron2

After installing the `detectron2` package, we require several small modifications to the library in order to speed-up evaluation procedures. To identify `detectron2`'s installation folder, you can use `pip show detectron2` (example output is: `anaconda/envs/d2/lib/python3.8/site-packages/detectron2`).

**Speeding up the evaluation by increasing batch size.** Open the `data/build.py` file (e.g `$CONDA_ENV_PATH$/site-packages/detectron2/data/build.py`). Inside, modify the `def build_detection_test_loader` function as follows:
- add `batch_size` argument to the function definition (with a default value of `batch_size=1` for backward compatibility)
- in the final `return` statement, replace `batch_size=1` with `batch_size=batch_size`

This way we add support of the variable batch size for the evaluation.

### VisualGenome experiments

To run VisualGenome experiments, we require further modifications to the Detectron2 source code. In LVIS + VisualGenome setup we support supervised training phase with mask annotations from LVIS (to train class-agnostic mask head). However, VisualGenome annotations (used for evaluation) don't contain masks. This causes issues during data loading, as some images contain masks, and some don't. To support such case we require modification of `datasets/lvis.py` (e.g. `$CONDA_ENV_PATH$/site-packages/detectron2/datasets/lvis.py`), line 150, by adding an `if` statement to check if segmentation mask is present in a given annotation. Specifically, please insert the following `if` statement to line 150 and place the segmentations-related logic under the new `if` as follows:

```python
if "segmentation" in anno:
    segm = anno["segmentation"]  # list[list[float]]
    ...
    obj["segmentation"] = segm
for extra_ann_key in extra_annotation_keys:  # <--- leave this line and the rest lines below untouched
    ...
```

## Download datasets

### COCO + LVIS experiments

For COCO + LVIS experiments, first download COCO images + annotations and LVIS annotations as instructed in the [Detectron2 data documentation](https://detectron2.readthedocs.io/en/latest/tutorials/builtin_datasets.html). For COCO, please download **COCO 2017** data with standard train/val annotations.

After downloading and extracting the standard datasets, please download the additional annotations for the COCO<sub>half</sub> dataset introduced in our paper: [coco_half_train.json](https://drive.google.com/file/d/1abBlME4Z5ggBUsmeO6o_trWvCrLzn7p1/view?usp=share_link) and [coco_half_val.json](https://drive.google.com/file/d/1G4pvMNJHJ9w37sTYqONafAtowSYecSi0/view?usp=share_link). Place them in the `coco/annotations/` folder. If you want to check how this dataset is registered by Detectron2, please see `configs/data/register_coco_half.py` script for details. This script is automatically called during training.

### LVIS + VisualGenome experiments

As LVIS and VisualGenome datasets largely overlap (50K images) and VisualGenome does not provide a default train-val split, for our work we devised a specific split. We selected our split so that the validation images for LVIS + VisualGenome setup are a subset of LVIS validation images. Specifically, we use LVIS validation images that can also be found in VisualGenome. For more details, please see our paper (supplementary).

First, download COCO and LVIS datasets as instructed above. Then, download the VisualGenome images v1.2 from the [official website](https://visualgenome.org/api/v0/api_home.html) and put all images **in a `visualgenome/` subfolder of a folder where COCO images are located**. By default, that would correspond to `$DETECTRON2_DATASETS$/coco/visualgenome/`, so an example image path would be `$DETECTRON2_DATASETS$/coco/visualgenome/2386299.jpg`.

For annotations, we had to pre-process official VisualGenome annotations to match the Detectron2 format. Please, download our custom annotation files [visualgenome_train.json](https://drive.google.com/file/d/1cqAXbSPXq0JrwxVWNkNbhIqwJql6tBgG/view?usp=share_link) and [visualgenome_val.json](https://drive.google.com/file/d/1XQKl2FOQY6pn6h-lh7QwN0baTalzgIs3/view?usp=share_link) and put them to the `$DETECTRON2_DATASETS$/visualgenome/` folder (if you want to use a different folder, please modify `configs/data/register_vglvis.py` file accordingly).

The final structure of your `DETECTRON2_DATASETS` folder should be the following:
- `coco/`
  - `annotations/`
    - `coco_half_train.json`
    - `coco_half_val.json`
    - `..original coco annotations (optional)..`
  - `train2017/`
  - `val2017/`
  - `visualgenome/`
- `lvis/`
  - `lvis_v1_val.json`
  - `lvis_v1_train.json`
- `visualgenome/`
  - `visualgenome_train.json`
  - `visualgenome_val.json`

## Download checkpoints

### Self-supervised backbone initialization

During the first, supervised phase, we initialize R-CNN from weights pretrained in a self-supervised manner, specifically, MoCo v2. In the table in the next sub-section, we provide weights that we downloaded in early 2022 as well. However, if you'd like to obtain the weights from the official MoCo v2 repository, please follow the next paragraph.

To obtain the original MoCo v2 weights, please download 800-epoch weights from MoCo official repository from [this table](https://github.com/facebookresearch/moco#models). Then, clone the [official moco repository](https://github.com/facebookresearch/moco), and [run a script as described here](https://github.com/facebookresearch/moco/tree/main/detection) to convert the weights to the Detectron2 format.

### Pretrained model weights

Below we provide weights for some of the trained models. For more details on the models' hyperparameters, please see our manuscript. Unfortunately, we cannot provide the weights for the original network described in the manuscript, but to reproduce our results, we have trained another network for 7500 iterations, and provide its weights. It can surpass the scores mentioned in the paper and achieves **7.35** mAP on all classes.

| Description | Link | Scores |
| --- | --- | --- |
| Pretrained MoCo v2 ResNet [official moco repository](https://github.com/facebookresearch/moco) used for backbone initialization | [`moco_v2_800ep_pretrain.pkl`](https://drive.google.com/file/d/1GzslI_Npk197QVwvfGfdONf91YeE9pXh/view?usp=sharing) | N/A |
| Fully-supervised Mask-RCNN with FPN and Res50 backbone, trained on COCO<sub>half</sub> dataset, used as the initialization for the discovery phase | [`supervised_cocohalf_maskrcnn.pth`](https://drive.google.com/file/d/1SRpHQBcCGz3cHPniFKq6yk1U5-mOOyOA/view?usp=share_link) | mAP<sub>COCO<sub>half</sub></sub>: 35.69 |
| Fully-supervised Mask-RCNN with FPN and Res50 backbone, trained on LVIS dataset, used as the initialization for the discovery phase | [`supervised_lvis_maskrcnn.pth`](https://drive.google.com/file/d/1OtQs_9UADQGbGvJPQnc8fXrpenqMguFQ/view?usp=share_link) | mAP<sub>LVIS</sub>: 18.47 |
| RNCDL network trained for discovery mode on COCO<sub>half</sub> and the rest of unlabeled images with the number of unlabeled classes set to 3000 | [`discovery_cocohalf_lvis_maskrcnn_lr3e-2_iter7500.pth`](https://drive.google.com/file/d/1vhFFNZSjpBbfWn-QjdRk9YW6eGFD7yRm/view?usp=share_link) | mAP<sub>COCO<sub>half</sub></sub>: 24.46, mAP<sub>LVIS</sub>: 5.94, **mAP<sub>all</sub>: 7.35** |

# Running experiments

We use [Detectron2's lazy configuration](https://detectron2.readthedocs.io/en/latest/tutorials/lazyconfigs.html) to define our framework. The configurations are located in `configs/` folder and the root config files can be found in `config/train/`.

For convenience, we provide [slurm](https://slurm.schedmd.com/documentation.html) example job scripts that are based on the scripts that we used to execute the jobs. They can be found in `slurm_scripts/`.
All our experiments were tested on 4 NVIDIA A40 GPUs with 48G memory.

To train our fully-supervised baselines, please use scripts in the `slurm_scripts/fully_supervised/` folder. E.g. to train a fully-supervised R-CNN on COCO<sub>half</sub> use:

```
DETECTRON2_DATASETS=/path/to/datasets \
PYTHONPATH=$PYTHONPATH:`pwd` \
python tools/train_supervised.py \
    \
    --config-file ./configs/train/fully_supervised/coco50pct.py \
    --num-gpus 4 \
    --dist-url 'tcp://localhost:10042' \
    \
    train.init_checkpoint=./checkpoints/moco_v2_800ep_pretrain.pkl \
    train.output_dir=./output \
    train.exp_name="cocohalf-supervised" \
    train.eval_period=20000
```

To train our discovery networks, please use scripts in the `slurm_scripts/discovery/` folder. E.g. to run discovery training for COCO<sub>half</sub> + LVIS setup use:

```
DETECTRON2_DATASETS=/path/to/datasets \
PYTHONPATH=$PYTHONPATH:`pwd` \
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
    train.init_checkpoint=./checkpoints/supervised_cocohalf_maskrcnn.pth \
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

To run discovery training for LVIS + VisualGenome setup, modify the command above as follows:

```
...
python tools/train_discovery.py \
    \
    --config-file ./configs/train/discovery/lvis_visualgenome.py \
    ...
    train.init_checkpoint=./checkpoints/supervised_lvis_maskrcnn.pth \
    ...
    model_supervised.roi_heads.box_predictor.discovery_model.num_unlabeled=5000 \
    ...
```

## Evaluation issues

Sometimes, after successfully finishing the discovery training phase and saving the weights, the script hangs or crashes during evaluation.

First, note that **discovery evaluation may take up to 2-3 hours**, so please make sure you give your script enough time to complete.

If you cannot obtain the results after several hours, you may cancel your job, and re-use the training script above with the dumped weights. For that, re-run the script above with the model weights set to your newly generated checkpoint (called `model_final.pth` by default in Detectron2),) `train.max_iter` set to 1, learning rate and weight decay set to 0, and `train.weights_mode="from_discovery"`. With such configuration, the script will load the saved weights and proceed to the evaluation. E.g.:

```
...
python tools/train_discovery.py \
    ...
    train.init_checkpoint="discovery_checkpoint.pth" \
    ...
    train.max_iter=1 \
    optimizer.lr=0.0 \
    optimizer.weight_decay=0.0 \
    train.weights_mode="from_discovery"
```

Even after this modification, **evaluation for the discovery part may take several hours**.

# Citation

If you find RNCDL useful in your research or reference it in your work, please star our repository and use the folowing:
```
@inproceedings{fomenko2022learning,
    author = {Vladimir Fomenko and Ismail Elezi and Deva Ramanan and Laura Leal-Taix{'e} and Aljo\v{s}a O\v{s}ep},
    title = {Learning to Discover and Detect Objects},
    booktitle={Advances in Neural Information Processing Systems},
    year = {2022}
}
```
