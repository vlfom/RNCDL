import torch

from typing import Dict, List
from detectron2.modeling.meta_arch import GeneralizedRCNN
from detectron2.modeling.roi_heads import CascadeROIHeads
from detectron2.modeling.roi_heads.fast_rcnn import fast_rcnn_inference


class GTPredictionExtractionRCNN(GeneralizedRCNN):
    """
    Generates class-predictions for provided GT annotations when `forward()` is called. Does not use RPN, perform NMS,
    etc, but simply applies classification RoI head onto the provided bboxes.

    Note: must use `FastRCNNDiscoveryOutputLayers` as its `roi_heads.box_predictor`.
    """

    def forward(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        """
        Receives a list of D2-formatted dicts of {images, annotations}, generates classification predictions for each
        of the annotation, and returns a Tensor with predictions.

        Returns: a tuple of:
            - Tensor of shape (N,), where N is the total number of annotations across all images,
            - None (for compatibility with `GeneralizedRCNN.inference()`).
        """

        images = self.preprocess_image(batched_inputs)  # Normalize the image
        features = self.backbone(images.tensor)  # Extract the FPN features

        anns = [x["instances"].to(self.device) for x in batched_inputs]  # Prepare annotations

        features = [features[f] for f in self.roi_heads.box_in_features]  # Extract features per FPN level

        box_features = self.roi_heads.box_pooler(  # RoI pooling
            features,
            [x.gt_boxes for x in anns]
        )
        box_features = self.roi_heads.box_head(box_features)
        predictions = self.roi_heads.box_predictor(box_features)  # Classification & localization preds

        # Get class predictions
        class_scores, _ = predictions
        _, class_pred = class_scores.max(1)

        return class_pred
