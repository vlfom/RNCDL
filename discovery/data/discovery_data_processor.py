import copy
import torch
import numpy as np
import random
import torchvision.transforms as T
import detectron2.data.transforms as D2_T
import torch.distributed as dist
import pycocotools.mask as mask_util

from PIL import ImageFilter
from detectron2.data.transforms import Augmentation, NoOpTransform, ColorTransform
from detectron2.data import (
    detection_utils as utils,
    DatasetMapper,
)
from detectron2.structures import (
    Boxes,
    BoxMode,
    Instances,
    BitMasks,
)


class RotateColorChannels:
    """Transforms BGR -> RGB or RGB -> BGR."""

    def __call__(self, x):
        return x[:, :, ::-1]


class ToNumpyArray:
    def __call__(self, x):
        # print("inside ToNumpyArray", type(x), x.size)
        return np.array(x)


class RandomImageTransform(Augmentation):
    def __init__(self, op, prob=0.5):
        super().__init__()
        self._init(locals())

    def get_transform(self, image):
        do = self._rand_range() < self.prob
        if do:
            return ColorTransform(self.op)
        else:
            return NoOpTransform()


class DiscoverTargetTransform:
    def __init__(self, mapping):
        self.mapping = mapping

    def __call__(self, y):
        y = self.mapping[y]
        return y


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[0.1, 2.0]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


class DiscoveryDataProcessor:

    def __init__(self, augmentations, num_augmentations, image_format):
        self.mapper = DiscoveryDatasetMapper(
            augmentations=augmentations,
            num_augmentations=num_augmentations,
            image_format=image_format,
            is_train=None,  # unused, kept for compatibility
        )

    def map_batched_data(self, ps_per_image):
        """
        Returns: list[tuple[dict]] of the length `num_augmentations`, where each inner tuple has the fixed length
                 of the number of proposals/annotations.
        """
        batched_inputs = []
        for p in ps_per_image:
            processed_input = self.mapper(p)
            batched_inputs.append(processed_input)
        batched_inputs = list(zip(*batched_inputs))
        return batched_inputs

    @staticmethod
    def strong_augmentations_list_swav_no_scalejitter():
        return [
            D2_T.ResizeShortestEdge(short_edge_length=(640, 672, 704, 736, 768, 800), sample_style="choice", max_size=1333,),
            D2_T.RandomFlip(horizontal=True),
            RandomImageTransform(T.ToPILImage(), prob=1.0),  # Convert to PIL to support applying torchvision's transforms
            RandomImageTransform(T.ColorJitter(0.8, 0.8, 0.8, 0.2), prob=0.8),
            RandomImageTransform(T.Grayscale(num_output_channels=3), prob=0.2),
            RandomImageTransform(GaussianBlur([0.1, 2.0]), prob=0.5),
            RandomImageTransform(ToNumpyArray(), prob=1.0),  # Convert PIL image back to numpy array
        ]


class DiscoveryDatasetMapper(DatasetMapper):

    def __init__(
        self,
        is_train,
        *,
        augmentations,
        num_augmentations,
        image_format,
    ):
        super().__init__(is_train, augmentations=augmentations, image_format=image_format)
        self.num_augmentations = num_augmentations

    def __call__(self, dataset_dict):
        """Loads & transforms the image and its annotations. Returns image content, proposals/annotations, and their categories.

        Args:
            dataset_dict (dict): Metadata of one image. Expected format (k: v):
              - "file_name": str,
              - "height": int,
              - "width": int,
              - "proposal_boxes": np.array of shape (N, 4), where N is the total number of proposals per image,
              - "proposal_bbox_mode": object of type BoxMode,
              - "proposal_categories": np.array of shape (N,).

        Returns:
            list[dict]: each item contains contains transformed image and its proposals/annotations, the length of
            the list is equal to the number of augmentations used in the pipeline
        """

        dataset_dict_original = dataset_dict  # Rename for cloning in the future

        # Load image
        image_original = utils.read_image(dataset_dict["file_name"], format=self.image_format)
        utils.check_image_size(dataset_dict, image_original)

        processed_dicts = []
        for i in range(self.num_augmentations):
            dataset_dict = copy.deepcopy(dataset_dict_original)  # Will be modified by code below
            image = image_original.copy()  # Clone the loaded image to use for the current round of augmentations

            # Instantiate transformations and augment image
            aug_input = D2_T.AugInput(image)
            transforms = self.augmentations(aug_input)  # Transforms `aug_input.image` in-place

            image = aug_input.image  # Get the transformed image
            image_shape = image.shape[:2]  # h, w

            # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
            # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
            # Therefore it's important to use torch.Tensor.
            dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))

            # We are in proposals transformation mode
            if "proposal_boxes" in dataset_dict:
                # Apply transforms to bboxes
                boxes = transforms.apply_box(
                    BoxMode.convert(
                        np.array(dataset_dict.pop("proposal_boxes")),  # D2 accepts only numpy arrays
                        dataset_dict.pop("proposal_bbox_mode"),
                        BoxMode.XYXY_ABS,
                    )
                )
                boxes = Boxes(torch.from_numpy(boxes))  # Convert to Boxes object

                # Save transformed proposals
                transformed_proposals = Instances(image_shape)
                transformed_proposals.proposal_boxes = boxes
                dataset_dict["proposals"] = transformed_proposals

                processed_dicts.append(dataset_dict)

            # We are in instances transformation mode
            elif "instances" in dataset_dict:
                # Apply transforms to bboxes
                boxes = [p["bbox"] for p in dataset_dict["instances"]]
                boxes = transforms.apply_box(
                    BoxMode.convert(
                        np.array(boxes),     # D2 accepts only numpy arrays
                        dataset_dict.pop("instances_bbox_mode"),
                        BoxMode.XYXY_ABS,
                    )
                )
                boxes = Boxes(torch.from_numpy(boxes))  # Convert to Boxes object

                # Apply transforms to segmentations
                segms = []
                for p in dataset_dict["instances"]:
                    segm = p["segmentation"]
                    mask = mask_util.decode(segm)
                    mask = transforms.apply_segmentation(mask)
                    assert tuple(mask.shape[:2]) == image_shape
                    segms.append(mask)
                segms = BitMasks(
                    torch.stack([torch.from_numpy(np.ascontiguousarray(x)) for x in segms])
                )

                # Extract categories
                classes = [p["category_id"] for p in dataset_dict["instances"]]
                classes = torch.tensor(classes, dtype=torch.int64)

                # Put all into instances
                target = Instances(image_shape)
                target.gt_boxes = boxes
                target.gt_classes = classes
                target.gt_masks = segms

                dataset_dict["instances"] = target

                processed_dicts.append(dataset_dict)

        return processed_dicts
