import time
import torch
import numpy as np
import torch.distributed as dist
import logging

from discovery.modeling.meta_arch.unified_rcnn_discoverer import ForwardMode

from detectron2.structures import BoxMode
from detectron2.evaluation.coco_evaluation import instances_to_coco_json
from detectron2.engine import AMPTrainer
from detectron2.utils import comm


class AMPDiscoveryTrainer(AMPTrainer):

    def __init__(
            self,
            model,
            full_optimizer,
            predictor_optimizer,
            supervised_train_loader,
            proposals_loader,
            discovery_data_processor,
            grad_scaler=None,
            supervised_loss_lambda=0.0,
            unlabeled_data_backbone_grad_coeff=0.0,
            detach_features_for_discovery=False,
    ):
        """
        Args:
            model, optimizer, grad_scaler: same as in :class:`AMPTrainer`.
            supervised_loss_lambda: coefficient to use for the supervised loss (relative to discovery loss).
        """

        # Use proposals_loader as the main dataloader, accessible via `data_loader`, `_data_loader_iter`
        super().__init__(model, proposals_loader, full_optimizer, grad_scaler)

        assert supervised_loss_lambda > 0, unlabeled_data_backbone_grad_coeff > 0

        # Set optimizers
        self.full_optimizer = full_optimizer
        self.predictor_optimizer = predictor_optimizer

        # Set up an additional supervised dataloader (default is the discovery one)
        self.supervised_data_loader = supervised_train_loader
        self._supervised_data_loader_iter = iter(supervised_train_loader)

        self.discovery_data_processor = discovery_data_processor
        self.supervised_loss_lambda = supervised_loss_lambda
        self.unlabeled_data_backbone_grad_coeff = unlabeled_data_backbone_grad_coeff

        self.detach_features_for_discovery = detach_features_for_discovery

        self._backbone_was_frozen = False
        self._backbone_was_unfrozen = False

        self._debug_dumped = 0

    def run_step(self):
        assert torch.cuda.is_available(), "[AMPTrainer] CUDA is required for AMP training!"
        from torch.cuda.amp import autocast

        start = time.perf_counter()
        discovery_data = next(self._data_loader_iter)
        supervised_data = next(self._supervised_data_loader_iter)
        data_time = time.perf_counter() - start

        with autocast():
            _is_discovery_memory_filled = self.model.module.is_discovery_network_memory_filled()

            # Collect supervised loss
            supervised_loss_dict = self._supervised_forward_get_loss(supervised_data)
            supervised_loss_dict = {"supervised_" + k: v for k, v in supervised_loss_dict.items()}
            supervised_loss_dict["supervised_loss"] = sum(supervised_loss_dict.values())

            # If the discovery network is still filling the memory, do not start the supervised training as well
            if not _is_discovery_memory_filled:
                supervised_loss_dict["supervised_loss"] *= 0

            # Generate discovery proposals (for a new batch of images)
            discovery_proposals = self._get_proposals(discovery_data)

            # Collect discovery loss
            # Perform an additional heads-only forward only if backbone is unfrozen
            discovery_loss_dict = self._discovery_forward_get_loss(
                discovery_proposals, extra_forward_for_backbone=False,
                detach_features_for_discovery=self.detach_features_for_discovery,
            )
            discovery_loss_dict = {"discovery_" + k: v for k, v in discovery_loss_dict.items()}

            loss = (
                    supervised_loss_dict["supervised_loss"] * self.supervised_loss_lambda +
                    discovery_loss_dict["discovery_loss"]
            )

        self.full_optimizer.zero_grad()  # Reset all gradients anyway

        # Backprop
        self.grad_scaler.scale(loss).backward()

        self.grad_scaler.step(self.predictor_optimizer)
        self.grad_scaler.update()

        loss_dict_all = {"total_loss_real": loss, **supervised_loss_dict, **discovery_loss_dict}
        self._write_metrics(loss_dict_all, data_time)

    def _supervised_forward_get_loss(self, batched_inputs):
        supervised_loss_dict = self.model(batched_inputs, mode=ForwardMode.SUPERVISED_TRAIN)
        return supervised_loss_dict

    def _get_proposals(self, batched_inputs):
        """
        Returns: list[dict], where each dict contains a list of proposals per image and has the following keys:
                 "image_id", "file_name", "width", "height", "proposal_boxes", "proposal_bbox_mode";
                 "proposal_boxes" contains a list[dict] with list of proposals per image and has the keys:
                 "image_id", "category_id", "bbox", "score".
        """

        with torch.no_grad():
            outputs = self.model(batched_inputs, mode=ForwardMode.PROPOSALS_EXTRACTION)

        proposals_per_image = []
        for image, output in zip(batched_inputs, outputs):

            # Format predictions to match the COCO format
            instances = output["instances"].to("cpu")
            instances_coco = instances_to_coco_json(instances, image["image_id"])
            prop_bboxes = [inst["bbox"] for inst in instances_coco]
            prop_segm = [inst["segmentation"] for inst in instances_coco]

            predictions = {
                "image_id": image["image_id"],
                "file_name": image["file_name"],
                "width": image["width"],
                "height": image["height"],
                "proposal_boxes": prop_bboxes,
                "proposal_bbox_mode": BoxMode.XYWH_ABS,
                "proposal_segm": prop_segm,
            }

            proposals_per_image.append(predictions)

        return proposals_per_image

    def _discovery_forward_get_loss(
            self, discovery_proposals, extra_forward_for_backbone=False, detach_features_for_discovery=False
    ):
        # Load data and apply augmentations
        batched_inputs = self.discovery_data_processor.map_batched_data(discovery_proposals)

        # Extract features for discovery proposals
        discovery_features = [
            self.model(view, mode=ForwardMode.DISCOVERY_FEATURE_EXTRACTION)
            for view in batched_inputs
        ]

        # Get discovery losses
        if detach_features_for_discovery:
            discovery_features = [f.detach() for f in discovery_features]

        discovery_losses = self.model(discovery_features, mode=ForwardMode.DISCOVERY_CLASSIFIER)

        if extra_forward_for_backbone:
            discovery_features_nograd = [f.detach() for f in discovery_features]
            discovery_losses_head_only = self.model(discovery_features_nograd, mode=ForwardMode.DISCOVERY_CLASSIFIER)
            discovery_losses["loss_heads_only"] = discovery_losses_head_only["loss"]

        return discovery_losses
