# Differences from the fully-supervised mode:
# - use class-agnostic R-CNN model
# - use COCO datasets for training/eval & COCO evaluator
# - use different #classes
# - use higher `test_score_thresh` for predictions (irrelevant for the future pipeline, just kept for COCO eval)

from detectron2.config import LazyCall as L
from detectron2.evaluation import COCOEvaluator

from configs.train.fully_supervised.lvis import lr_multiplier, optimizer, train, dataloader
from configs.models.mask_rcnn_fpn_discovery import model # 이 부분도 고쳐야할 것 같음 (faster r-cnn 부분 models에 따로 구현)

# Update datasets for training and validation
from configs.data.register_voc_half import voc_meta  # Register coco_50pct train dataset

dataloader.train.dataset.names = "voc_half_train"
dataloader.test.dataset.names = "voc_half_val"

# Update the evaluator
dataloader.evaluator = L(COCOEvaluator)(
    dataset_name="${..test.dataset.names}",
)
model.roi_heads.box_predictor.test_score_thresh = 0.05

# Update the number of classes
model.roi_heads.num_classes = 20
model.roi_heads.box_predictor.num_classes = 20
model.roi_heads.box_predictor.discovery_model.num_labeled = 20 + 1

# Set bbox localization head to be class-agnostic
model.roi_heads.box_predictor.cls_agnostic_bbox_reg = True

# Set mask head to be class-agnostic
model.roi_heads.mask_head.num_classes = 1