import os
from pathlib import Path
from torch.utils.data import Dataset

from ImageCache import ImageCache


class TestDataset(Dataset):
    """
    Dataset for .jpg images and labels
    """

    def __init__(self, root: str):
        self._search_for_files(root)
        self.cache = ImageCache(force_rgb=True)

    def __getitem__(self, index):
        filename = self.image_files[index]
        label = self.image_labels[filename]

        img = self.cache.load(filename)

        return img, label

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
