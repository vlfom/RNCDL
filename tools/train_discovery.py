#!/usr/bin/env python

import logging
import wandb
import torch.distributed as dist
import torch.nn as nn

from detectron2.solver.build import maybe_add_gradient_clipping
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import LazyConfig, instantiate, CfgNode, LazyCall as L
from detectron2.engine import (
    default_argument_parser,
    default_setup,
    default_writers,
    hooks,
    launch,
)
from detectron2.engine.defaults import create_ddp_model
from detectron2.evaluation import print_csv_format, inference_on_dataset
from detectron2.utils import comm

from discovery.modeling.meta_arch.unified_rcnn_discoverer import DiscoveryRCNN, ForwardMode
from discovery.engine.discovery_train_loop import AMPDiscoveryTrainer
from discovery.engine.discovery_inference import discovery_inference_on_dataset

logger = logging.getLogger("detectron2")


def do_discovery_test(cfg, model):
    # Result on the supervised data
    model.module.set_default_forward_mode(ForwardMode.SUPERVISED_INFERENCE)
    ret = inference_on_dataset(
        model,
        instantiate(cfg.supervised_dataloader.test),
        instantiate(cfg.supervised_dataloader.evaluator),
    )
    model.module.remove_default_forward_mode()
    print_csv_format(ret)

    # Result on the discovery data
    ret = discovery_inference_on_dataset(
        model,
        instantiate(cfg.discovery_test_loader_knowns),
        instantiate(cfg.discovery_test_loader),
        instantiate(cfg.discovery_class_mapping_evaluator),
        instantiate(cfg.discovery_evaluator),
    )
    print_csv_format(ret)

    return ret


def do_train(args, cfg):
    logger = logging.getLogger("detectron2")

    # Instantiate supervised model
    model_supervised = instantiate(cfg.model_supervised)

    # Try load weights for supervised model if training from scratch
    if cfg.train.weights_mode == "from_supervised":
        DetectionCheckpointer(model_supervised, cfg.train.output_dir).resume_or_load(
            cfg.train.init_checkpoint, resume=False
        )

    # Instantiate the meta-model and other downstream models
    model = DiscoveryRCNN(
        supervised_rcnn=model_supervised,
        cfg=cfg,
    )

    # Load weights for the full model if resuming training
    if cfg.train.weights_mode == "from_discovery":
        DetectionCheckpointer(model, cfg.train.output_dir).resume_or_load(
            cfg.train.init_checkpoint, resume=True
        )

    # Assert the models reference the same backbone
    assert model.supervised_rcnn.backbone is model.proposals_extractor_rcnn.backbone
    assert model.supervised_rcnn.backbone is model.discovery_feature_extractor.backbone
    assert model.supervised_rcnn.backbone is model.discovery_gt_class_predictions_extractor.backbone

    model.to(cfg.train.device)
    logger.info("Model:\n{}".format(model))

    model = create_ddp_model(model, **cfg.train.ddp)

    # Set up full optimizer
    if hasattr(cfg.optimizer, "params"):
        cfg.optimizer.params.model = model
    else:
        cfg.optimizer.model = model

    full_optim = instantiate(add_gradient_clipping(cfg.optimizer, cfg.train.enable_gradient_clipping))

    # Set up box_predictor optimizer
    box_predictor = model.module.supervised_rcnn.roi_heads.box_predictor
    if isinstance(box_predictor, nn.ModuleList):
        box_predictor = box_predictor[0]
    if hasattr(cfg.optimizer, "params"):
        cfg.optimizer.params.model = box_predictor.discovery_model
    else:
        cfg.optimizer.model = box_predictor.discovery_model

    predictor_optim = instantiate(add_gradient_clipping(cfg.optimizer, cfg.train.enable_gradient_clipping))

    # Set up data loaders/processors
    supervised_train_loader = instantiate(cfg.supervised_dataloader.train)
    proposals_loader = instantiate(cfg.proposals_dataloader)
    discovery_data_processor = instantiate(cfg.discovery_data_processor)

    # Define trainer
    trainer = AMPDiscoveryTrainer(
        model,
        full_optim,
        predictor_optim,
        supervised_train_loader, proposals_loader,
        discovery_data_processor,
        supervised_loss_lambda=cfg.train.supervised_loss_lambda,
        unlabeled_data_backbone_grad_coeff=0,
        detach_features_for_discovery=False,
    )

    # Auxiliary
    checkpointer = DetectionCheckpointer(
        model,
        cfg.train.output_dir,
        full_optim=full_optim,
        predictor_optim=predictor_optim,
        trainer=trainer,
    )
    trainer.register_hooks(
        [
            hooks.IterationTimer(),
            hooks.LRScheduler(scheduler=instantiate(cfg.lr_multiplier)),
            hooks.PeriodicCheckpointer(checkpointer, **cfg.train.checkpointer)
            if comm.is_main_process()
            else None,
            hooks.EvalHook(cfg.train.eval_period, lambda: do_discovery_test(cfg, model)),
            hooks.PeriodicWriter(
                default_writers(cfg.train.output_dir, cfg.train.max_iter),
                period=cfg.train.log_period,
            )
            if comm.is_main_process()
            else None,
        ]
    )

    # Train
    start_iter = 0
    trainer.train(start_iter, cfg.train.max_iter)


def add_gradient_clipping(optimizer, clip=False):
    if not clip:
        return optimizer

    optim_cfg = CfgNode({
        "SOLVER": {
            "CLIP_GRADIENTS": {
                "ENABLED": True,
                "CLIP_TYPE": "value",
                "CLIP_VALUE": 1.0,
                "NORM_TYPE": 2.0,
            }
        }
    })
    optimizer = L(maybe_add_gradient_clipping)(
        cfg=optim_cfg,
        optimizer=optimizer
    )
    return optimizer


def main(args):
    cfg = LazyConfig.load(args.config_file)
    cfg = LazyConfig.apply_overrides(cfg, args.opts)
    default_setup(cfg, args)

    if dist.get_rank() == 0:
        _mode = "online" if cfg.train.use_wandb else "disabled"
        wandb.init(project=cfg.train.wandb_project_name, name=cfg.train.exp_name, config=args, mode=_mode)

    do_train(args, cfg)


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
