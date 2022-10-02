import torch
import torch.nn as nn

from typing import Dict, List, Optional
from detectron2.structures import ImageList, Instances
from detectron2.modeling.meta_arch import GeneralizedRCNN
from detectron2.modeling.roi_heads import StandardROIHeads, CascadeROIHeads


class FeatureExtractionRCNN(GeneralizedRCNN):

    def forward(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        """
        Receives a list of D2-formatted dicts of {images, annotations, proposals}, extracts features
        for each of the proposals via FPN, and returns a Tensor with extracted features.

        Returns: a tuple of:
            - Tensor of shape (M, 256 * 7 * 7), where M is the total number of proposals across all images,
            - None (for compatibility with `GeneralizedRCNN.inference()`).
        """

        images = self.preprocess_image(batched_inputs)  # Normalize the image
        features = self.backbone(images.tensor)  # Extract the FPN features

        assert "proposals" in batched_inputs[0]

        proposals = [x["proposals"].to(self.device) for x in batched_inputs]  # Extract pre-supplied proposals (no RPN)
        bbox_features = self.roi_heads(images, features, proposals)  # Extract features per proposal bbox

        return bbox_features


class FeatureExtractionROIHeadsWrapper(nn.Module):

    def __init__(self, roi_heads):
        super().__init__()

        self.roi_heads = roi_heads

    def forward(
            self,
            images: ImageList,
            features: Dict[str, torch.Tensor],
            proposals: List[Instances],
            targets: Optional[List[Instances]] = None,
    ):
        """Based on the parent's `forward()`. All arguments are kept for compatibility.

        Returns: a Tensor of shape (M, channel_dim * output_size * output_size), i.e. (M, 256 * 7 * 7),
        where M is the total number of proposals aggregated over all N batch images.
        See: https://github.com/facebookresearch/detectron2/blob/a24729abd7a08aa29b453e1754779004807fc8ee/detectron2/modeling/poolers.py#L195
        """

        # `CascadeROIHeads`-based
        if isinstance(self.roi_heads, CascadeROIHeads):
            features = [features[f] for f in self.roi_heads.box_in_features]  # Extract features per FPN level
            prev_pred_boxes = None
            image_sizes = [x.image_size for x in proposals]

            for k in range(self.roi_heads.num_cascade_stages):
                if k > 0:
                    proposals = self._create_proposals_from_boxes(prev_pred_boxes, image_sizes)

                if k < self.roi_heads.num_cascade_stages - 1:
                    predictions = self._run_stage(features, proposals, k)
                    prev_pred_boxes = self.roi_heads.box_predictor[k].predict_boxes(predictions, proposals)
                else:
                    # At the last stage we don't need actual predictions, but only features
                    box_features = self.roi_heads.box_pooler(features, [x.proposal_boxes for x in proposals])
                    box_features = self.roi_heads.box_head[k](box_features)

        # `StandardROIHeads`-based
        elif isinstance(self.roi_heads, StandardROIHeads):
            features = [features[f] for f in self.roi_heads.box_in_features]  # Extract features per FPN level
            box_features = self.roi_heads.box_pooler(  # RoI pooling
                features,
                [x.proposal_boxes for x in proposals]
            )
            box_features = self.roi_heads.box_head(box_features)

        else:
            raise ValueError("self.roi_heads must be an instance of StandardROIHeads or CascadeROIHeads")

        return box_features

    def forward_with_given_boxes(self, features: Dict[str, torch.Tensor], instances: List[Instances]):
        return self.roi_heads.forward_with_given_boxes(features, instances)

    def _forward_box(self, features: Dict[str, torch.Tensor], proposals: List[Instances]):
        return self.roi_heads._forward_box(features, proposals)

    def _forward_mask(self, features: Dict[str, torch.Tensor], instances: List[Instances]):
        return self.roi_heads._forward_box(features, instances)

    def _forward_keypoint(self, features: Dict[str, torch.Tensor], instances: List[Instances]):
        return self.roi_heads._forward_keypoint(features, instances)

    def _match_and_label_boxes(self, proposals, stage, targets):
        return self.roi_heads._match_and_label_boxes(proposals, stage, targets)

    def _run_stage(self, features, proposals, stage):
        return self.roi_heads._run_stage(features, proposals, stage)

    def _create_proposals_from_boxes(self, boxes, image_sizes):
        return self.roi_heads._create_proposals_from_boxes(boxes, image_sizes)
