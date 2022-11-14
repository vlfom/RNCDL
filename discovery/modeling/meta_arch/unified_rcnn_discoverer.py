import torch
import torch.nn as nn

from detectron2.modeling.roi_heads import StandardROIHeads
from detectron2.modeling.meta_arch import GeneralizedRCNN

from discovery.modeling.meta_arch.gt_prediction_extraction_rcnn import GTPredictionExtractionRCNN
from discovery.modeling.meta_arch.feature_extraction_rcnn import FeatureExtractionRCNN, FeatureExtractionROIHeadsWrapper
from discovery.modeling.roi_heads.class_agnostic_fast_rcnn_discovery_output_layers_wrapper import ClassAgnosticFastRCNNDiscoveryOutputLayersWrapper


class ForwardMode:
    SUPERVISED_TRAIN = 0
    SUPERVISED_INFERENCE = 1
    PROPOSALS_EXTRACTION = 2
    DISCOVERY_FEATURE_EXTRACTION = 3
    DISCOVERY_CLASSIFIER = 4
    DISCOVERY_GT_CLASS_PREDICTIONS_EXTRACTION = 5
    DISCOVERY_INFERENCE = 6


class DiscoveryRCNN(nn.Module):

    def __init__(
        self,
        *,
        supervised_rcnn,
        cfg
    ):
        super().__init__()

        self.supervised_rcnn = supervised_rcnn
        self.remove_bkg_class_from_discovery_model()

        proposals_extractor_rcnn = init_proposals_extraction_model(supervised_rcnn, cfg)
        discovery_feature_extractor = init_discovery_feature_extractor_model(supervised_rcnn)
        discovery_gt_class_predictions_extractor = init_discovery_gt_prediction_extractor_model(supervised_rcnn)

        self.proposals_extractor_rcnn = proposals_extractor_rcnn
        self.discovery_feature_extractor = discovery_feature_extractor
        self.discovery_gt_class_predictions_extractor = discovery_gt_class_predictions_extractor

        self.eval_knowns_param = cfg.eval_knowns_param
        self.eval_all_param = cfg.eval_all_param

        self.num_known_classes = self.supervised_rcnn.roi_heads.num_classes

        self._default_forward_mode = None

        self.discovery_nms_thresh = cfg.model_proposals_extraction_param.test_nms_thresh

    def forward(self, x, mode=None):
        if mode is None:
            if self._default_forward_mode:
                mode = self._default_forward_mode
            else:
                raise ValueError("Forward mode must be specified or the default one must be set")

        if mode == ForwardMode.SUPERVISED_TRAIN:
            return self._forward_supervised_train(x)
        elif mode == ForwardMode.SUPERVISED_INFERENCE:
            return self._forward_supervised_inference(x)
        elif mode == ForwardMode.PROPOSALS_EXTRACTION:
            return self._forward_proposals_extractor(x)
        elif mode == ForwardMode.DISCOVERY_FEATURE_EXTRACTION:
            return self._forward_discovery_feature_extractor(x)
        elif mode == ForwardMode.DISCOVERY_CLASSIFIER:
            return self._forward_discovery_classifier(x)
        elif mode == ForwardMode.DISCOVERY_GT_CLASS_PREDICTIONS_EXTRACTION:
            return self._forward_discovery_gt_class_predictions_extractor(x)
        elif mode == ForwardMode.DISCOVERY_INFERENCE:
            return self._forward_discovery_inference(x)
        else:
            raise ValueError(f"Unknown forward mode: {mode}")

    def _forward_supervised_train(self, batched_inputs):
        self.supervised_rcnn.train()

        if not isinstance(self.supervised_rcnn.roi_heads.box_predictor, nn.ModuleList):  # baseline, non-centernet
            self.supervised_rcnn.proposal_generator.nms_thresh = 0.7
        else:
            self.supervised_rcnn.proposal_generator.nms_thresh_train = 0.9
            self.supervised_rcnn.proposal_generator.nms_thresh_test = 0.9

        return self.supervised_rcnn(batched_inputs)

    def _forward_supervised_inference(self, batched_inputs):
        self.supervised_rcnn.eval()

        if not isinstance(self.supervised_rcnn.roi_heads.box_predictor, nn.ModuleList):  # baseline, non-centernet
            self.supervised_rcnn.proposal_generator.nms_thresh = 0.7
        else:
            self.supervised_rcnn.proposal_generator.nms_thresh_train = 0.9
            self.supervised_rcnn.proposal_generator.nms_thresh_test = 0.9

        if isinstance(self.supervised_rcnn.roi_heads.box_predictor, nn.ModuleList):
            box_predictors = self.supervised_rcnn.roi_heads.box_predictor
        else:
            box_predictors = [self.supervised_rcnn.roi_heads.box_predictor]

        for box_predictor in box_predictors:
            box_predictor.allow_novel_classes_during_inference = False
            box_predictor.test_topk_per_image = self.eval_knowns_param["test_topk_per_image"]
            box_predictor.test_score_thresh = self.eval_knowns_param["test_score_thresh"]

        return self.supervised_rcnn(batched_inputs)

    def _forward_proposals_extractor(self, batched_inputs):
        self.proposals_extractor_rcnn.eval()

        if not isinstance(self.supervised_rcnn.roi_heads.box_predictor, nn.ModuleList):  # baseline, non-centernet
            self.proposals_extractor_rcnn.proposal_generator.nms_thresh = self.discovery_nms_thresh
        else:
            self.supervised_rcnn.proposal_generator.nms_thresh_train = self.discovery_nms_thresh
            self.supervised_rcnn.proposal_generator.nms_thresh_test = self.discovery_nms_thresh

        results = self.proposals_extractor_rcnn(batched_inputs)
        return results

    def _forward_discovery_feature_extractor(self, batched_inputs):
        self.discovery_feature_extractor.eval()
        return self.discovery_feature_extractor(batched_inputs)

    def _forward_discovery_classifier(self, features):
        box_predictor = self.supervised_rcnn.roi_heads.box_predictor
        if isinstance(box_predictor, nn.ModuleList):
            box_predictor = box_predictor[0]

        box_predictor.discovery_model.train()

        return box_predictor.discovery_model(features)

    def _forward_discovery_gt_class_predictions_extractor(self, batched_inputs):
        self.discovery_gt_class_predictions_extractor.eval()

        if isinstance(self.discovery_gt_class_predictions_extractor.roi_heads.box_predictor, nn.ModuleList):
            box_predictors = self.discovery_gt_class_predictions_extractor.roi_heads.box_predictor
        else:
            box_predictors = [self.discovery_gt_class_predictions_extractor.roi_heads.box_predictor]

        for box_predictor in box_predictors:
            box_predictor.allow_novel_classes_during_inference = True

        results = self.discovery_gt_class_predictions_extractor(batched_inputs)
        return results

    def _forward_discovery_inference(self, batched_inputs):
        self.supervised_rcnn.eval()

        if not isinstance(self.supervised_rcnn.roi_heads.box_predictor, nn.ModuleList):  # baseline, non-centernet
            # self.supervised_rcnn.proposal_generator.nms_thresh = self.discovery_nms_thresh
            self.supervised_rcnn.proposal_generator.nms_thresh = 0.9
        else:
            # self.supervised_rcnn.proposal_generator.nms_thresh_test = self.discovery_nms_thresh
            self.supervised_rcnn.proposal_generator.nms_thresh_train = 0.9
            self.supervised_rcnn.proposal_generator.nms_thresh_test = 0.9

        if isinstance(self.supervised_rcnn.roi_heads.box_predictor, nn.ModuleList):
            box_predictors = self.supervised_rcnn.roi_heads.box_predictor
        else:
            box_predictors = [self.supervised_rcnn.roi_heads.box_predictor]

        for box_predictor in box_predictors:
            box_predictor.allow_novel_classes_during_inference = True
            box_predictor.test_topk_per_image = self.eval_all_param["test_topk_per_image"]
            box_predictor.test_score_thresh = self.eval_all_param["test_score_thresh"]

        results = self.supervised_rcnn(batched_inputs)
        return results

    def is_discovery_network_memory_filled(self):
        box_predictor = self.supervised_rcnn.roi_heads.box_predictor
        if isinstance(box_predictor, nn.ModuleList):
            box_predictor = box_predictor[0]

        is_memory_filled = box_predictor.discovery_model.memory_patience == 0
        return is_memory_filled

    def remove_bkg_class_from_discovery_model(self):
        if isinstance(self.supervised_rcnn.roi_heads.box_predictor, nn.ModuleList):
            box_predictors = self.supervised_rcnn.roi_heads.box_predictor
        else:
            box_predictors = [self.supervised_rcnn.roi_heads.box_predictor]

        with torch.no_grad():
            for box_predictor in box_predictors:
                box_predictor.discovery_model.head_lab.weight = nn.Parameter(
                    box_predictor.discovery_model.head_lab.weight[:-1]
                )
                box_predictor.discovery_model.head_lab.bias = nn.Parameter(
                    box_predictor.discovery_model.head_lab.bias[:-1]
                )


    def set_default_forward_mode(self, mode):
        self._default_forward_mode = mode

    def remove_default_forward_mode(self):
        self._default_forward_mode = None


def init_proposals_extraction_model(model_supervised, cfg):
    # Modify the network's RoI output head; new one performs class-agnostic localization adjustments and
    # no classification

    roi_heads = StandardROIHeads(
        num_classes=model_supervised.roi_heads.num_classes,
        batch_size_per_image=model_supervised.roi_heads.batch_size_per_image,
        positive_fraction=model_supervised.roi_heads.positive_fraction,
        proposal_matcher=model_supervised.roi_heads.proposal_matcher,
        box_in_features=model_supervised.roi_heads.box_in_features,
        box_pooler=model_supervised.roi_heads.box_pooler,
        box_head=model_supervised.roi_heads.box_head,
        box_predictor=ClassAgnosticFastRCNNDiscoveryOutputLayersWrapper(
            box_predictor=model_supervised.roi_heads.box_predictor,
            min_box_size=cfg.model_proposals_extraction_param.min_box_size,
            test_nms_thresh=cfg.model_proposals_extraction_param.test_nms_thresh,
            test_topk_per_image=cfg.model_proposals_extraction_param.test_topk_per_image,
        ),
        mask_in_features=model_supervised.roi_heads.mask_in_features,
        mask_pooler=model_supervised.roi_heads.mask_pooler,
        mask_head=model_supervised.roi_heads.mask_head,
    )

    model_proposals_extraction = GeneralizedRCNN(
        backbone=model_supervised.backbone,
        proposal_generator=model_supervised.proposal_generator,
        roi_heads=roi_heads,
        pixel_mean=model_supervised.pixel_mean,
        pixel_std=model_supervised.pixel_std,
        input_format=model_supervised.input_format,
    )
    return model_proposals_extraction


def init_discovery_feature_extractor_model(model_supervised):
    model_discovery_feature_extractor = FeatureExtractionRCNN(
        backbone=model_supervised.backbone,
        proposal_generator=None,
        roi_heads=FeatureExtractionROIHeadsWrapper(
            roi_heads=model_supervised.roi_heads
        ),
        pixel_mean=model_supervised.pixel_mean,
        pixel_std=model_supervised.pixel_std,
        input_format=model_supervised.input_format,
    )
    return model_discovery_feature_extractor


def init_discovery_gt_prediction_extractor_model(model_supervised):
    model_gt_extraction = GTPredictionExtractionRCNN(
        backbone=model_supervised.backbone,
        proposal_generator=None,
        roi_heads=model_supervised.roi_heads,
        pixel_mean=model_supervised.pixel_mean,
        pixel_std=model_supervised.pixel_std,
        input_format=model_supervised.input_format,
    )
    return model_gt_extraction
