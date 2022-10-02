# Modifications to the default COCO setup:
# - use RGB input format
# - use RFS
# - train longer & use LVIS-specific schedule
# - use more # classes
# - use lower `test_score_thresh` for predictions
# - use LVIS dataset for training
# - use LVIS evaluator

from detectron2.config import LazyCall as L
from detectron2.evaluation import LVISEvaluator
from detectron2.data.samplers import RepeatFactorTrainingSampler

from configs.util.train import train
from configs.util.optim import SGD as optimizer
from configs.data.coco import dataloader
from configs.util.coco_schedule import lr_multiplier_3x as lr_multiplier
from configs.models.mask_rcnn_fpn_discovery import model

from discovery.data.discovery_data_processor import DiscoveryDataProcessor


# Use AMP
train.amp.enabled = True
train.ddp.fp16_compression = True
train.ddp.find_unused_parameters = True

# Switch to RGB mode (as per instructions in the MoCo repo)
dataloader.train.mapper.image_format = "RGB"
dataloader.test.mapper.image_format = "RGB"

# Use RFS
dataloader.train.sampler = L(RepeatFactorTrainingSampler)(
    repeat_factors=L(RepeatFactorTrainingSampler.repeat_factors_from_category_frequency)(
        dataset_dicts="${...dataset}",
        repeat_thresh=0.001
    )
)

# Wandb
train.use_wandb = True
train.wandb_project_name = "thesis-fall"

###
### LVIS-specific config
###

# Train longer
train.max_iter = 180000
lr_multiplier.scheduler.milestones = [120000, 160000, 180000]

# Update the number of classes
model.roi_heads.num_classes = 1203
model.roi_heads.box_predictor.num_classes = 1203
model.roi_heads.box_predictor.discovery_model.num_labeled = 1203 + 1

# Update datasets
dataloader.train.dataset.names = "lvis_v1_train"
dataloader.test.dataset.names = "lvis_v1_val"

# Use LVIS evaluator and 300 detections per image
dataloader.evaluator = L(LVISEvaluator)(
    dataset_name="lvis_v1_val",
    distributed=True,
    max_dets_per_image=300
)
model.roi_heads.box_predictor.test_score_thresh = 0.0001
