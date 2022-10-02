from detectron2.config import LazyCall as L
from detectron2.evaluation import LVISEvaluator

from .lvis import train, dataloader, lr_multiplier, model, optimizer
from configs.data.register_vglvis import image_root

# Train longer
train.max_iter = 180000
lr_multiplier.scheduler.milestones = [120000, 160000, 180000]

# Update the number of classes
model.roi_heads.num_classes = 3929
model.roi_heads.box_predictor.num_classes = 3929
model.roi_heads.box_predictor.discovery_model.num_labeled = 3929 + 1

# Update datasets
dataloader.train.dataset.names = "vg_lvis_train"
dataloader.test.dataset.names = "vg_lvis_val"

# Use LVIS evaluator and 300 detections per image
dataloader.evaluator = L(LVISEvaluator)(
    dataset_name="vg_lvis_val",
    distributed=True,
    max_dets_per_image=300
)

# Delete mask head (no mask annotations)
del model.roi_heads.mask_head
del model.roi_heads.mask_in_features
del model.roi_heads.mask_pooler
