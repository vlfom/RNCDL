import torch.nn.functional as F
import torch

from detectron2.layers import cat, cross_entropy, nonzero_tuple
from detectron2.modeling.roi_heads import FastRCNNOutputLayers
from detectron2.modeling.roi_heads.fast_rcnn import _log_classification_stats, fast_rcnn_inference


class FastRCNNDiscoveryOutputLayers(FastRCNNOutputLayers):

    def __init__(
        self,
        input_shape,
        *,
        box2box_transform,
        num_classes,
        test_score_thresh=0.0,
        test_nms_thresh=0.5,
        test_topk_per_image=100,
        cls_agnostic_bbox_reg=False,
        smooth_l1_beta=0.0,
        box_reg_loss_type="smooth_l1",
        loss_weight=1.0,
        discovery_model=None,
        allow_novel_classes_during_training=False,
        allow_novel_classes_during_inference=False,
    ):
        super().__init__(
            input_shape,
            box2box_transform=box2box_transform,
            num_classes=num_classes,
            test_score_thresh=test_score_thresh,
            test_nms_thresh=test_nms_thresh,
            test_topk_per_image=test_topk_per_image,
            cls_agnostic_bbox_reg=cls_agnostic_bbox_reg,
            smooth_l1_beta=smooth_l1_beta,
            box_reg_loss_type=box_reg_loss_type,
            loss_weight=loss_weight
        )

        self.discovery_model = discovery_model

        # Remove linear layer for classification - a discovery cosine classifier will be used instead
        del self.cls_score

        self.allow_novel_classes_during_training = allow_novel_classes_during_training
        self.allow_novel_classes_during_inference = allow_novel_classes_during_inference

        # During supervised pretraining we use outputs of knowns head only
        if not allow_novel_classes_during_training:
            self.cls_score = self.cls_score_discovery_knowns_bg_only

        # Use concatenated logits from both knowns and novels heads
        else:
            self.cls_score = self.cls_score_discovery

    def cls_score_discovery_knowns_bg_only(self, x):
        logits = self.discovery_model.forward_knowns_bg_head_single_view(x)
        return logits

    def cls_score_discovery(self, x):
        # logits_full, logits_full_over = self.discovery_model.forward_heads_single_view(x)
        # return logits_full, logits_full_over
        logits_full = self.discovery_model.forward_heads_single_view(x)
        return logits_full

    def losses(self, predictions, proposals):
        if not self.allow_novel_classes_during_training:
            return super().losses(predictions, proposals)

        # In discovery mode, a previously used notion of "background" class is no longer relevant - such classes
        # might be novel ones; therefore, we calculate losses only for foreground (matched) proposals
        # Note: for bbox_regression it is the default behavior, so we update code only for classification loss

        # (scores, scores_over), proposal_deltas = predictions
        scores, proposal_deltas = predictions

        # Get GTs
        gt_classes = (cat([p.gt_classes for p in proposals], dim=0) if len(proposals) else torch.empty(0))
        _log_classification_stats(scores, gt_classes)

        # Calculate classification loss (only for matched proposals)
        fg_inds = nonzero_tuple((gt_classes >= 0) & (gt_classes < self.num_classes))[0]
        # loss_cls = (
        #    cross_entropy(scores[fg_inds], gt_classes[fg_inds], reduction="mean") +
        #    cross_entropy(scores_over[fg_inds], gt_classes[fg_inds], reduction="mean")
        # ) / 2
        loss_cls = cross_entropy(scores[fg_inds], gt_classes[fg_inds], reduction="mean")

        # fake_loss = 0
        # for param in self.discovery_model.head_unlab_over.parameters():
        #     fake_loss += torch.sum(param)
        # loss_cls += 0 * fake_loss

        # Calculate localization loss
        if len(proposals):
            proposal_boxes = cat([p.proposal_boxes.tensor for p in proposals], dim=0)  # Nx4
            assert not proposal_boxes.requires_grad, "Proposals should not require gradients!"
            gt_boxes = cat(
                [(p.gt_boxes if p.has("gt_boxes") else p.proposal_boxes).tensor for p in proposals],
                dim=0,
            )
        else:
            proposal_boxes = gt_boxes = torch.empty((0, 4), device=proposal_deltas.device)
        loss_box = self.box_reg_loss(
            proposal_boxes, gt_boxes, proposal_deltas, gt_classes
        )

        losses = {
            "loss_cls": loss_cls,
            "loss_box_reg": loss_box
        }
        return {k: v * self.loss_weight.get(k, 1.0) for k, v in losses.items()}

    def predict_probs(self, predictions, proposals):
        if not self.allow_novel_classes_during_training:
            return super().predict_probs(predictions, proposals)

        scores, _ = predictions
        probs = F.softmax(scores, dim=-1)

        # During inference, we sometimes wants to predict only known classes; also we need to add a fake "background"
        # class that exists in the default R-CNN implementation
        if not self.training:

            # In this mode, we want to evaluate performance on labeled dataset with known classes only; therefore,
            # we have to keep predictions for knowns only
            if not self.allow_novel_classes_during_inference:
                # Keep probabilities for known (+ fake background) classes only
                probs = probs[:, :self.num_classes + 1]

            # In this mode, we want to evaluate performance on all classes; for that we also need to create a fake
            # logit for "background" class as `fast_rcnn_inference` ignores the last element
            else:
                # Append fake logits for a background class
                probs = torch.cat(
                    [probs, torch.zeros((probs.shape[0], 1), dtype=probs.dtype).to(probs.device)],
                    axis=1
                )

        num_inst_per_image = [len(p) for p in proposals]
        probs = probs.split(num_inst_per_image, dim=0)
        return probs
