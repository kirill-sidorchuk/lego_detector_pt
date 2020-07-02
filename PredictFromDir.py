import argparse
import os

import cv2
import numpy as np
import torch
from torch import nn

from ImageDataset import ImageDataset
from train import load_model


def get_tensor_batch(images: list):
    channels = []
    img_height = images[0].shape[0]
    img_width = images[0].shape[1]
    for img in images:
        channels.append(cv2.split(img.astype(np.float32) / 255))
    return {'rgb': torch.Tensor(channels).view(len(images), 3, img_height, img_width)}


def predict_with_tta(tta: int, images: list, model: nn.Module, target_size: tuple, tta_mode='mean'):
    tta_hflip = tta > 0
    tta_vflip = tta > 1
    tta_rotate = tta > 2

    images_batch = []
    for _image in images:
        image = cv2.resize(_image, target_size, interpolation=cv2.INTER_AREA)
        images_batch.append(image)
        augment(image, images_batch, tta_hflip, tta_rotate, tta_vflip)

    batch = get_tensor_batch(images_batch)
    with torch.set_grad_enabled(False):
        probs = model.forward(batch)

    return aggregate_ensemble_probs(probs, tta_mode)


def aggregate_ensemble_probs(probs_all: torch.Tensor, tta_mode: str) -> np.ndarray:
    probs_all = probs_all.cpu().numpy()
    if tta_mode == 'mean':
        return np.mean(probs_all, axis=0)
    else:
        raise Exception('Not implemented')


def augment(image: np.ndarray, tta_batch: list, tta_hflip: bool, tta_rotate: bool, tta_vflip: bool):
    if tta_hflip:
        tta_batch.append(cv2.flip(image, 0))
    if tta_vflip:
        tta_batch.append(cv2.flip(image, 1))
    if tta_rotate:
        tta_batch.append(rotate(image))


def rotate(img: np.ndarray) -> np.ndarray:
    width = img.shape[1]
    height = img.shape[0]
    angle = np.random.rand() * 45
    scale = 1 + np.random.rand() * 1.1
    M = cv2.getRotationMatrix2D((width / 2, height / 2), angle, scale)
    return cv2.warpAffine(img, M, (width, height), flags=cv2.INTER_AREA, borderMode=(cv2.BORDER_REFLECT_101))


def predict_from_dir(args):

    images = os.listdir(args.dir)
    if len(images) == 0:
        print('Cannot find any images in ' + args.dir)
        return
    print('%d images to classify found' % len(images))

    use_cuda = torch.cuda.is_available() and args.gpu.lower() == 'true'
    device = torch.device("cuda" if use_cuda else "cpu")
    print("Using GPU" if use_cuda else "Using CPU")

    # loading labels
    labels = ImageDataset.load_labels(args.data_root)
    if labels is None:
        print("Failed to load labels. Data root is " + args.data_root)
        return

    num_classes = len(labels)
    print("Number of recognized classes = %d" % num_classes)

    # loading model
    print("loading model...")
    model, model_name = load_model(args.model, device, num_classes, inference=True)
    target_size = (224, 224)

    # loading snapshot
    snapshot_file = os.path.join("snapshots", model_name, "snapshot_%d.pt" % args.snapshot)
    if not os.path.exists(snapshot_file):
        print("Snapshot file does not exists: " + snapshot_file)
        return
    model.load_state_dict(torch.load(snapshot_file, 'cpu'))

    for image_name in images:
        img = cv2.imread(os.path.join(args.dir, image_name))
        if img is None:
            print('Failed to load ' + image_name)
            continue

        img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)

        # getting required number of frames for prediction
        images_to_predict = [img]

        probs = predict_with_tta(args.tta, images_to_predict, model, target_size)
        sorted_indexes = np.argsort(probs)
        for t in range(5):
            label_index = sorted_indexes[-t - 1]
            label_prob = probs[label_index]
            label_name = labels[label_index]
            s = "%s: %1.2f%% %s" % (image_name, label_prob * 100.0, label_name)
            print(s)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run prediction from web camera')
    parser.add_argument("model", type=str, default=None, help="Name of the model to train.")
    parser.add_argument("--data_root", type=str, default=None, help="Path to data.")
    parser.add_argument("--snapshot", type=int, default=None, help="Iteration of a snapshot.")
    parser.add_argument("--gpu", type=str, default='true', help="Set to 'true' for GPU.")
    parser.add_argument("--dir", type=str, help="directory to scan for images")
    parser.add_argument("--tta", type=int, default=0, help="0 - no TTA, 1 - hflip, 2 - vflip+hflip")
    parser.add_argument("--rtta", type=int, default=0,
                        help="Robot TTA. <1 - no TTA, >1 - number of images to take for TTA")
    parser.add_argument("--tta_mode", type=str, default="mean", help="'mean' or 'majority' voting TTA")

    _args = parser.parse_args()
    predict_from_dir(_args)
