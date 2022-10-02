import datetime
import logging
import time
import torch
import detectron2.utils.comm as comm
import numpy as np

from detectron2.utils.logger import log_every_n_seconds

from discovery.modeling.meta_arch.unified_rcnn_discoverer import ForwardMode


@torch.no_grad()
def discovery_inference_on_dataset(model, knowns_data_loader, data_loader, evaluator_class_mapping, evaluator_discovery):
    evaluator_class_mapping.reset()
    evaluator_discovery.reset()

    # Generate category mapping (predicted_id -> LVIS_id)
    model.module.set_default_forward_mode(ForwardMode.DISCOVERY_GT_CLASS_PREDICTIONS_EXTRACTION)
    _run_generic_evaluation_loop(model, knowns_data_loader, evaluator_class_mapping)
    _run_generic_evaluation_loop(model, data_loader, evaluator_class_mapping)
    model.module.remove_default_forward_mode()

    class_mapping = evaluator_class_mapping.evaluate()

    comm.synchronize()
    class_mapping = comm.all_gather(class_mapping)  # Share class mapping across GPUs
    class_mapping = list(filter(lambda x: x is not None, class_mapping))[0]  # We got a single non-None

    # Evaluate on discovery dataset
    evaluator_discovery.set_class_mapping(class_mapping)

    model.module.set_default_forward_mode(ForwardMode.DISCOVERY_INFERENCE)
    _run_generic_evaluation_loop(model, data_loader, evaluator_discovery)
    model.module.remove_default_forward_mode()

    results = evaluator_discovery.evaluate()

    if results is None:
        results = {}

    return results


def _run_generic_evaluation_loop(model, data_loader, evaluator):
    logger = logging.getLogger(__name__)
    logger.info("###\n### Start discovery inference on {} batches\n###".format(len(data_loader)))

    total = len(data_loader)  # inference data loader must have a fixed length
    num_warmup = min(5, total - 1)

    start_time = time.perf_counter()
    for idx, inputs in enumerate(data_loader):
        if idx == num_warmup:
            start_time = time.perf_counter()

        outputs = model(inputs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        evaluator.process(inputs, outputs)

        iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
        total_seconds_per_iter = (time.perf_counter() - start_time) / iters_after_start
        if idx >= num_warmup * 2:
            eta = datetime.timedelta(seconds=int(total_seconds_per_iter * (total - idx - 1)))
            log_every_n_seconds(
                logging.INFO,
                (
                    f"Inference done {idx + 1}/{total}. "
                    f"Total: {total_seconds_per_iter:.4f} s/iter. "
                    f"ETA={eta}"
                ),
                n=5,
            )

    # Measure the time only for this worker (before the synchronization barrier)
    total_time = time.perf_counter() - start_time
    total_time_str = str(datetime.timedelta(seconds=total_time))
    # NOTE this format is parsed by grep
    logger.info("Total inference time: {} ({:.6f} s / iter per device)".format(
        total_time_str, total_time / (total - num_warmup)
    ))