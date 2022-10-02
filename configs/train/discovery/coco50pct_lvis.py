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

from ..fully_supervised.coco50pct import (
    optimizer,
    dataloader as supervised_dataloader,
    train,
    model as model_supervised,
)

# Reduce the LR (best for discovery heads when using batch_size=32; ablated separately)
optimizer.lr = 0.01

# Set the number of novel classes
model_supervised.roi_heads.box_predictor.discovery_model.num_unlabeled = 2100

# Set memory
model_supervised.roi_heads.box_predictor.discovery_model.memory_batches = 100
model_supervised.roi_heads.box_predictor.discovery_model.memory_patience = 150

# No unused parameters anymore
train.ddp.find_unused_parameters = False

# Use novels heads during supervised training as well
model_supervised.roi_heads.box_predictor.allow_novel_classes_during_training = True

# Use all proposals that got matched to GT annotations (no sampling)
model_supervised.roi_heads.positive_fraction = 1.0

# Define the proposals extraction model parameters
model_proposals_extraction_param = dict(
    test_nms_thresh=1.01,
    test_topk_per_image=50,
    min_box_size=0,
)

# Define lambda for supervised loss
train.supervised_loss_lambda = 0.01

# Define training duration
num_epochs = 10
one_epoch_niter = int(100170 / 16)
train.max_iter = num_epochs * one_epoch_niter

# Use cosine LR schedule
lr_multiplier = L(WarmupParamScheduler)(
    scheduler=L(CosineParamScheduler)(
        start_value=1,
        end_value=0.1,
    ),
    warmup_length=0.5 * one_epoch_niter / train.max_iter,
    warmup_method="linear",
    warmup_factor=1e-3,
)

# Define proposals extractor dataloader (on LVIS images)
proposals_dataloader = L(build_detection_train_loader)(
    dataset=L(get_detection_dataset_dicts)(names="lvis_v1_train", filter_empty=False),
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
model_supervised.roi_heads.box_predictor.discovery_model.batch_size = 4  # 4 per GPU

# Define discovery data processor
discovery_data_processor = L(DiscoveryDataProcessor)(
    augmentations=L(DiscoveryDataProcessor.strong_augmentations_list_swav_no_scalejitter)(),
    num_augmentations=2,
    image_format="RGB"
)

# Discovery evaluation
# Note: `build_detection_test_loader` source code has to be modified in-place to allow for `batch_size` argument
discovery_test_loader = L(build_detection_test_loader)(
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

discovery_test_loader_knowns = L(build_detection_test_loader)(
    dataset=L(get_detection_dataset_dicts)(names="coco_half_val", filter_empty=False),
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
        dataset_name="lvis_v1_val",
        distributed=True,
        known_class_ids=[3, 12, 34, 35, 36, 41, 45, 58, 60, 76, 77, 80, 90, 94, 99, 118, 127, 133, 139, 154, 173, 183,
                         207, 217, 225, 230, 232, 271, 296, 344, 367, 378, 387, 421, 422, 445, 469, 474, 496, 534, 569,
                         611, 615, 631, 687, 703, 705, 716, 735, 739, 766, 793, 816, 837, 881, 912, 923, 943, 961, 962,
                         964, 976, 982, 1000, 1019, 1037, 1071, 1077, 1079, 1095, 1097, 1102, 1112, 1115, 1123, 1133,
                         1139, 1190, 1202],
    ),
    novel_class_id_thresh=10000,
)

# Evaluation parameters for COCO dataset
eval_knowns_param = {
    "test_topk_per_image": 100,
    "test_score_thresh": 0.05,
}

# Evaluation parameters for LVIS dataset
eval_all_param = {
    "test_topk_per_image": 300,
    "test_score_thresh": 0.0001,
}

# Weights loading mode
train.weights_mode = "from_supervised"
