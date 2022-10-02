import torch
import torch.nn as nn
import json

from typing import Dict, List


class LVISPredictionsLoader(nn.Module):
    """
    A class that is used as a dummy model that loads pre-generated predictions from the discovery phase and then
    returns them per given image when `forward()` is called.
    """

    def __init__(self, file_path):
        """Loads pre-generated predictions grouped by "image_id".

        Args:
            file_path: path to the pre-generated predictions file. The file contains a dict with keys corresponding
            to "image_id"s and values being list[dict], each dict with the keys: "image_id", "category_id",
            "bbox", "score", "segmentation" (matching the output format of ``instances_to_coco_json()``).
        """

        super().__init__()

        with open(file_path, "r") as f:
            self.image_predictions = json.loads(f.read())

        self.dummy_parameter = nn.Parameter(torch.zeros(1))

    def forward(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        """Returns pre-generated predictions for the given images.

        Args: same as ``GeneralizedRCNN.forward()``.
        Returns:
            list[dict]: each dict is the output for one input image. It matches the output format of
                        ``instances_to_coco_json()``.
        """

        predictions = []
        for inp in batched_inputs:
            im_id = inp["image_id"]

            # Add +1 to each of the predicted categories to undo the following category_id subtraction performed
            # by D2's `LVISEvaluator`. See evaluation/lvis_evaluation.py, line 154:
            # https://github.com/facebookresearch/detectron2/blob/e3053c14cfcce515de03d3093c0a7b4e734fa41f/detectron2/evaluation/lvis_evaluation.py#L154
            instances = self.image_predictions[str(im_id)]
            for inst in instances:
                inst["category_id"] -= 1

            predictions.append({
                "image_id": im_id,
                "instances": instances,
            })

        return predictions
