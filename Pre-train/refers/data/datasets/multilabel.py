from collections import defaultdict
import glob
import json
import os
from typing import Callable, List, Tuple

import cv2
import numpy as np
from torch.utils.data import Dataset

from refers.data.structures import ImageCaptionInstance, ImageCaptionBatch
from refers.data import transforms as T


class MultiLabelClassificationDataset(Dataset):
    r"""
    A dataset which provides image-labelset pairs from COCO instance annotation
    files. This is used for multilabel classification pretraining task.

    Parameters
    ----------
    data_root: str, optional (default = "datasets/coco")
        Path to the dataset root directory. This must contain images and
        annotations (``train2017``, ``val2017`` and ``annotations`` directories).
    split: str, optional (default = "train")
        Which split (from COCO 2017 version) to read. One of ``{"train", "val"}``.
    image_tranform: Callable, optional (default = refers.data.transforms.DEFAULT_IMAGE_TRANSFORM)
        A list of transformations, from either `albumentations
        <https://albumentations.readthedocs.io/en/latest/>`_ or :mod:`refers.data.transforms`
        to be applied on the image.
    """

    def __init__(
        self,
        data_root: str,
        split: str,
        image_transform: Callable = T.DEFAULT_IMAGE_TRANSFORM,
    ):
        self.image_transform = image_transform

        # Make a tuple of image id and its filename, get image_id from its
        # filename (assuming directory has images with names in COCO 2017 format).
        image_filenames = glob.glob(os.path.join(data_root, f"{split}2017", "*.jpg"))
        self.id_filename: List[Tuple[int, str]] = [
            (int(os.path.basename(name)[:-4]), name) for name in image_filenames
        ]
        # Load the instance (bounding box and mask) annotations.
        _annotations = json.load(
            open(
                os.path.join(data_root, "annotations", f"instances_{split}2017.json")
            )
        )
        # Make a mapping between COCO category id and its index, to make IDs
        # consecutive, else COCO has 80 classes with IDs 1-90. Start index from 1
        # as 0 is reserved for background (padding idx).
        _category_ids = {
            ann["id"]: index + 1
            for index, ann in enumerate(_annotations["categories"])
        }

        # A mapping between image ID and list of unique category IDs (indices as above)
        # in corresponding image.
        self.instances = defaultdict(list)

        for ann in _annotations["annotations"]:
            self.instances[ann["image_id"]].append(_category_ids[ann["category_id"]])

        # De-duplicate instances and drop empty labels, we only need to do
        # classification.
        self.instances = {
            image_id: list(set(ins))
            for image_id, ins in self.instances.items()
            if len(ins) > 0
        }
        # Filter out image IDs which didn't have any instances.
        self.id_filename = [
            (t[0], t[1]) for t in self.id_filename if t[0] in self.instances
        ]
        # Padding while forming a batch, because images may have variable number
        # of instances. We do not need padding index from tokenizer: COCO has
        # category ID 0 as background, conventionally.
        self.padding_idx = 0

    def __len__(self):
        return len(self.id_filename)

    def __getitem__(self, idx: int):
        # Get image ID and filename.
        image_id, filename = self.id_filename[idx]

        # Open image from path and apply transformation, convert to CHW format.
        image = cv2.imread(filename)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.image_transform(image=image)["image"]
        image = np.transpose(image, (2, 0, 1))

        # Get a list of instances present in the image.
        instances = self.instances[image_id]

        # Treat list of instances as "caption tokens" for reusability.
        # TODO (kd): it is hacky and written in deadline rush, make it better.
        return ImageCaptionInstance(image_id, image, caption_tokens=instances)

    def collate_fn(self, instances: List[ImageCaptionInstance]) -> ImageCaptionBatch:
        return ImageCaptionBatch(instances, padding_value=self.padding_idx)
