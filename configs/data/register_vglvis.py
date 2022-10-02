import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_lvis_json
from detectron2.data.datasets.builtin_meta import COCO_CATEGORIES, _get_coco_instances_meta

from ._meta_register_vglvis import VG_LVIS_CATEGORIES, VG_LVIS_CATEGORY_IMAGE_COUNT


_root = os.getenv("DETECTRON2_DATASETS", "datasets")

json_path_train = os.path.join(_root, "visualgenome/visualgenome_train.json")
json_path_val = os.path.join(_root, "visualgenome/visualgenome_val.json")
image_root = os.path.join(_root, "coco/")

def _get_vg_lvis_instances_meta():
    cat_ids = [k["id"] for k in VG_LVIS_CATEGORIES]
    assert min(cat_ids) == 1 and max(cat_ids) == len(
        cat_ids
    ), "Category ids are not in [1, #categories], as expected"
    # Ensure that the category list is sorted by id
    vg_lvis_categories = sorted(VG_LVIS_CATEGORIES, key=lambda x: x["id"])
    thing_classes = [k["synset"] for k in vg_lvis_categories]
    meta = {"thing_classes": thing_classes, "class_image_count": VG_LVIS_CATEGORY_IMAGE_COUNT}
    return meta

vg_lvis_meta = _get_vg_lvis_instances_meta()

# Check if the datasets were registered already (this may happen due to multiple imports in downstream files)
try:
    DatasetCatalog.get("vg_lvis_train")
except:
    DatasetCatalog.register(
        name="vg_lvis_train",
        func=lambda: load_lvis_json(
            json_file=json_path_train,
            image_root=image_root,
        )
    )
    MetadataCatalog.get("vg_lvis_train").set(
        json_file=json_path_train, image_root=image_root, evaluator_type="lvis", **vg_lvis_meta
    )

    DatasetCatalog.register(
        name="vg_lvis_val",
        func=lambda: load_lvis_json(
            json_file=json_path_val,
            image_root=image_root,
        )
    )
    MetadataCatalog.get("vg_lvis_val").set(
        json_file=json_path_val, image_root=image_root, evaluator_type="lvis", **vg_lvis_meta
    )

