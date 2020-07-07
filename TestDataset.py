import os
from pathlib import Path
from typing import Tuple

import numpy as np
from torch.utils.data import Dataset

from ImageCache import ImageCache
from ImageDataset import ImageDataset


class TestDataset(Dataset):
    """
    Dataset for .jpg images and labels
    """

    def __init__(self, root: str, test_dir: str):
        self._search_for_files(os.path.join(root, test_dir))
        self.cache = ImageCache(force_rgb=True)

        # loading labels
        self.labels = ImageDataset.load_labels(root)
        if self.labels is None:
            print("Failed to load labels. Data root is " + root)
            return

        # checking for extra classes
        labels_set = set(self.labels)
        filtered_files = []
        for filename in self.image_files:
            label = self.image_labels[filename]
            if label not in labels_set:
                # raise Exception('test label not in training set: ' + label)
                pass
            else:
                filtered_files.append(filename)

        if len(filtered_files) != len(self.image_files):
            print('Test files filtered to conform to training labels. %d files were excluded' % (
                        len(self.image_files) - len(filtered_files)))
            self.image_files = filtered_files

        # indexing labels
        self.label_to_index = dict()
        for i, label in enumerate(self.labels):
            self.label_to_index[i] = label

    def __getitem__(self, index) -> Tuple[np.ndarray, int]:
        """
        Returns test set image and its label.
        :param index: int index
        :return: tuple
        """
        filename = self.image_files[index]
        label = self.image_labels[filename]

        img = self.cache.load(filename)

        return img, self.label_to_index[label]

    def __len__(self):
        return len(self.image_files)

    def _search_for_files(self, root_dir: str):
        """
        Search for all image *.jpg files in a directory recuresively
        :param root_dir: directory to search in
        """

        self.image_files = []
        self.image_labels = {}
        for path in Path(root_dir).rglob('*.jpg'):
            file_path = str(path)
            base_dir = os.path.split(file_path)[0]
            label = os.path.split(base_dir)[1]
            self.image_files.append(file_path)
            self.image_labels[file_path] = label
