import torch
import torch.nn as nn

from typing import List, Tuple
from detectron2.structures import Boxes, Instances
from detectron2.layers import batched_nms


class ClassAgnosticFastRCNNDiscoveryOutputLayersWrapper(nn.Module):

    def __init__(
        self,
        box_predictor,
        min_box_size=0,
        test_nms_thresh=0.5,
        test_topk_per_image=50,
    ):
        super().__init__()

        self.box_predictor = box_predictor
        self.min_box_size = min_box_size
        self.test_nms_thresh = test_nms_thresh
        self.test_topk_per_image = test_topk_per_image

    def inference(self, predictions: Tuple[torch.Tensor, torch.Tensor], proposals: List[Instances]):
        """
        Args:
            predictions: return values of :meth:`forward()`.
            proposals (list[Instances]): proposals that match the features that were
                used to compute predictions. The ``proposal_boxes`` field is expected.

        Returns:
            - list[Instances]: list of localization-adjusted predictions with the following properties:
                * `pred_boxes`: Boxes object of shape (Ni, 4) with bbox coordinates in XYXY_ABS format;
                * `scores`: RPN objectness logits
                * `pred_classes`: 1-D Tensor of -1s (kept for compatibility)
            - None: placeholder for compatibility.
        """

        boxes = self.box_predictor.predict_boxes(predictions, proposals)  # Apply localization deltas
        image_shapes = [x.image_size for x in proposals]

        # Based off `fast_rcnn_inference_single_image()`.
        # See: https://github.com/facebookresearch/detectron2/blob/c081505af16f54c5a013916b81636d03f7c0a8fd/detectron2/modeling/roi_heads/fast_rcnn.py#L117.
        proposals_logits = [proposals_per_image.objectness_logits for proposals_per_image in proposals]
        results_per_image = extract_class_agnostic_proposals(
            boxes, image_shapes, proposals_logits,
            self.min_box_size, self.test_nms_thresh, self.test_topk_per_image
        )
        return results_per_image, None

    def forward(self, x):
        return self.box_predictor.forward(x)

    def losses(self, predictions, proposals):
        return self.box_predictor.losses(predictions, proposals)

    def box_reg_loss(self, proposal_boxes, gt_boxes, pred_deltas, gt_classes):
        return self.box_predictor.box_reg_loss(proposal_boxes, gt_boxes, pred_deltas, gt_classes)

    def predict_boxes_for_gt_classes(self, predictions, proposals):
        return self.box_predictor.predict_boxes_for_gt_classes(predictions, proposals)

    def predict_boxes(self, predictions, proposals):
        return self.box_predictor.predict_boxes(predictions, proposals)

    def predict_probs(self, predictions, proposals):
        return self.box_predictor.predict_probs(predictions, proposals)


def extract_class_agnostic_proposals(boxes, image_shapes, proposals_logits, min_box_size, nms_thresh, topk_per_image):
    results_per_image = []

    for boxes_per_image, image_shape, scores_per_image in zip(boxes, image_shapes, proposals_logits):

        valid_mask = torch.isfinite(boxes_per_image).all(dim=1)
        if not valid_mask.all():
            boxes_per_image = boxes_per_image[valid_mask]
            scores_per_image = scores_per_image[valid_mask]

        # Convert to Boxes to use the `clip` function ...
        boxes_per_image = Boxes(boxes_per_image.reshape(-1, 4))
        boxes_per_image.clip(image_shape)

        # Filter out small bboxes
        keep = boxes_per_image.nonempty(threshold=min_box_size-1e-8)
        boxes_per_image, scores_per_image = boxes_per_image[keep], scores_per_image[keep]

        boxes_per_image = boxes_per_image.tensor.view(-1, 4)  # R x 4, class agnostic

        # Apply class-agnostic NMS & keep topK
        keep = batched_nms(
            boxes_per_image,
            scores_per_image,
            torch.zeros(boxes_per_image.shape[0]),  # Mark all bboxes as same-class, to apply NMS for all at once
            nms_thresh
        )
        keep = keep[:topk_per_image]  # Keep topK only
        boxes_per_image, scores_per_image = boxes_per_image[keep], scores_per_image[keep]

        result = Instances(image_shape)
        result.pred_boxes = Boxes(boxes_per_image)
        result.scores = scores_per_image  # Put RPN logits instead of class probability
        result.pred_classes = -1 * result.scores.new_ones(result.scores.shape)  # Fill with -1s, for compatibility

        results_per_image.append(result)

    return results_per_image
