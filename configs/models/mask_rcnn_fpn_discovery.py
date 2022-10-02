# Adapt the standard Mask RCNN model to MOCO and to have the cosine RoI classification heads

from detectron2.config import LazyCall as L
from detectron2.layers import ShapeSpec
from detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.layers.batch_norm import NaiveSyncBatchNorm

from discovery.modeling.roi_heads.fast_rcnn_discovery_output_layers import FastRCNNDiscoveryOutputLayers
from discovery.modeling.discovery.discovery_network import DiscoveryClassifier

from .mask_rcnn_fpn import model

# Define discovery classification heads: 80 known classes + 1 bg class
discovery_heads = L(DiscoveryClassifier)(
    num_labeled=80 + 1,
    num_unlabeled=1,
    feat_dim=1024,
    hidden_dim=2048,
    proj_dim=256,
    num_views=2,
    memory_batches=100,
    items_per_batch=50,
    memory_patience=150,
    num_iters_sk=3,
    epsilon_sk=0.05,
    temperature=0.1,
    batch_size=4,
)

# Modify `box_predictor` classifier
model.roi_heads.box_predictor = L(FastRCNNDiscoveryOutputLayers)(
    input_shape=ShapeSpec(channels=1024),
    test_score_thresh=0.05,
    box2box_transform=L(Box2BoxTransform)(weights=(10, 10, 5, 5)),
    num_classes=80,
    discovery_model=discovery_heads,
    allow_novel_classes_during_training=False,
    allow_novel_classes_during_inference=False,
)

# Use SyncBN
model.backbone.bottom_up.stem.norm = \
    model.backbone.bottom_up.stages.norm = \
    model.backbone.norm = "SyncBN"
model.roi_heads.box_head.conv_norm = lambda c: NaiveSyncBatchNorm(c, stats_mode="N")
model.roi_heads.mask_head.conv_norm = lambda c: NaiveSyncBatchNorm(c, stats_mode="N")

# Prepare backbone for MoCo weights initialization (as per instructions in the MoCo repo)
model.backbone.bottom_up.freeze_at = 2
model.backbone.bottom_up.stages.stride_in_1x1 = False

# Switch to the RGB mode (as per instructions in the MoCo repo)
model.pixel_mean = [123.675, 116.280, 103.530]
model.pixel_std = [58.395, 57.12, 57.375]
model.input_format = "RGB"
