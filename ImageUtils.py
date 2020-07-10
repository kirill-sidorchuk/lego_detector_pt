import cv2
import numpy as np
import torch


def resize_to_resolution(im, downsample_size):
    if max(im.shape[0], im.shape[1]) > downsample_size:
        # downsampling
        if im.shape[0] > im.shape[1]:
            dsize = ((downsample_size * im.shape[1]) // im.shape[0], downsample_size)
        else:
            dsize = (downsample_size, (downsample_size * im.shape[0]) // im.shape[1])
        im = cv2.resize(im, dsize, interpolation=cv2.INTER_AREA)

    return im


def expand_mask(mask, kernel_size, n_iter=1):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.dilate(mask, kernel, iterations=n_iter)


def shrink_mask(mask, kernel_size, n_iter=1):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.erode(mask, kernel, iterations=n_iter)


def convert_to_8bpp(arr_float):
    return np.clip(arr_float, 0, 255).astype(np.uint8)


def image_to_tensor(img: np.ndarray, unsqueeze: bool) -> torch.Tensor:
    """
    Converts 8 bit BGR image [H, W, C] to 32-bit RGB float tensor [1, C, H, W]
    :param img: 8 bit BGR image to convert.
    :param unsqueeze: True to add extra batch dimension
    :return: torch tensor [1, C, H, W] float 32 [0..1]
    """
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = torch.from_numpy(img)
    img = img.permute(2, 0, 1)
    if unsqueeze:
        img = img.unsqueeze(0)
    img = img.to(torch.float32) / 255.
    return img