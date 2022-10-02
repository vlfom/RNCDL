import detectron2.data.transforms as T

from detectron2.solver import WarmupParamScheduler
from detectron2.config import LazyCall as L
from detectron2.data import (
    DatasetMapper,
    build_detection_train_loader,
    build_detection_test_loader,
    get_detection_dataset_dicts,
)
from detectron2.evaluation import LVISEvaluator
from fvcore.common.param_scheduler import CosineParamScheduler

from discovery.data.discovery_data_processor import DiscoveryDataProcessor
from discovery.evaluation.evaluator_discovery import ClassMappingEvaluator, DiscoveryEvaluator
from discovery.evaluation.lvis_evaluation import LVISEvaluatorDiscovery

from configs.data.register_vglvis import image_root
from .coco50pct_lvis import (
    optimizer, supervised_dataloader, train,
    model_proposals_extraction_param, lr_multiplier,
    proposals_dataloader, discovery_data_processor,
    discovery_test_loader, discovery_class_mapping_evaluator,
    discovery_evaluator, eval_knowns_param,
    eval_all_param,
    model_supervised,
)

supervised_dataloader.train.dataset.names = "lvis_v1_train"
supervised_dataloader.test.dataset.names = "lvis_v1_val"
supervised_dataloader.evaluator = L(LVISEvaluator)(
    dataset_name="lvis_v1_val",
    distributed=True,
    max_dets_per_image=300
)
model_supervised.roi_heads.box_predictor.test_score_thresh = 0.0001

# Update the number of classes
model_supervised.roi_heads.num_classes = 1203
model_supervised.roi_heads.box_predictor.num_classes = 1203
model_supervised.roi_heads.box_predictor.discovery_model.num_labeled = 1203 + 1

# Define proposals extractor dataloader (on LVIS images)
proposals_dataloader = L(build_detection_train_loader)(
    dataset=L(get_detection_dataset_dicts)(names="vg_lvis_train", filter_empty=False),
    mapper=L(DatasetMapper)(
        is_train=False,
        augmentations=[
            L(T.ResizeShortestEdge)(short_edge_length=800, max_size=1333),
        ],
        image_format="RGB",
    ),
    total_batch_size=16,   # For discovery phase, use a batch size of 16 = 4*4 per GPU\
    num_workers=4,
)

# Discovery evaluation
# Note: `build_detection_test_loader` source code has to be modified in-place to allow for `batch_size` argument
discovery_test_loader = L(build_detection_test_loader)(
    dataset=L(get_detection_dataset_dicts)(names="vg_lvis_val", filter_empty=False),
    mapper=L(DatasetMapper)(
        is_train=True,  # We must keep annotations to generate class mapping (predicted_id -> GT_id)
        augmentations=[
            L(T.ResizeShortestEdge)(short_edge_length=800, max_size=1333),
        ],
        image_format="RGB",
    ),
    batch_size=4,
    num_workers=4,
)

discovery_test_loader_knowns = L(build_detection_test_loader)(
    dataset=L(get_detection_dataset_dicts)(names="lvis_v1_val", filter_empty=False),
    mapper=L(DatasetMapper)(
        is_train=True,  # We must keep annotations to generate class mapping (predicted_id -> GT_id)
        augmentations=[
            L(T.ResizeShortestEdge)(short_edge_length=800, max_size=1333),
        ],
        image_format="RGB",
    ),
    batch_size=4,
    num_workers=4,
)

discovery_class_mapping_evaluator = L(ClassMappingEvaluator)(
    last_free_class_id=10000, max_class_num=20000,
)

discovery_evaluator = L(DiscoveryEvaluator)(
    evaluator=L(LVISEvaluatorDiscovery)(
        dataset_name="vg_lvis_val",
        distributed=True,
        known_class_ids=list(range(1, 1204)),
    ),
    novel_class_id_thresh=10000,
)

# Evaluation parameters for LVIS dataset
eval_knowns_param = {
    "test_topk_per_image": 300,
    "test_score_thresh": 0.0001,
}
