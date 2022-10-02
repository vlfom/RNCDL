from detectron2.evaluation import LVISEvaluator


class LVISModifiedEvaluator(LVISEvaluator):
    """
    Evaluator that introduces a minor extension to the standard LVIS evaluator.

    The only difference is that the predictions provided to this evaluator are already processed as in
    `instances_to_coco_json()` and can be directly evaluated.
    (this is due to predictions coming from the pre-generated files where they are preprocessed already)
    """

    def process(self, inputs, outputs):
        """
        Args:
            Same as `LVISEvaluator.process()`.
        """

        # Predictions are supplied by ``LVISPredictionsLoader`` which returns them in the COCO format already
        self._predictions += outputs