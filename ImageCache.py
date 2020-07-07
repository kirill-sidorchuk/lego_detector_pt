import cv2
import numpy as np
from typing import Tuple


class ImageCache:
    """
    Caches images into RAM. Image is loaded from disk only once. After loading image is kept in RAM in unpacked format.
    """

    def __init__(self, force_rgb: bool, dst_img_size: Tuple[int, int] or None = None):
        self.cache = {}
        self.forceRgb = force_rgb
        self.dst_img_size = dst_img_size

    def load(self, filename: str) -> np.ndarray:
        """
        Loads image from disk. If image is already in cache it is immediately returned. Else - it is loaded and placed
        in cache.
        :param filename: image file name path.
        :return: image as numpy array
        """

        if filename not in self.cache:
            img = cv2.imread(filename, cv2.IMREAD_COLOR if self.forceRgb else cv2.IMREAD_UNCHANGED)
            if img is None:
                raise Exception('Failed to load image: ' + filename)
            if self.dst_img_size is not None:
                img = cv2.resize(img, self.dst_img_size, interpolation=cv2.INTER_AREA)
            self.cache[filename] = img

        return self.cache[filename]
