import os

import torch
from torch import nn

import cv2
import numpy as np
import argparse

from ImageDataset import ImageDataset
from modelA import modelA
from modelB import modelB


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
        # counts = np.sum(to_categorical(np.argmax(probs_all, axis=1), num_classes=len(probs_all[0])), axis=0)
        # results.append(counts / np.sum(counts))


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


def clip(frame):
    width = frame.shape[1]
    height = frame.shape[0]
    x0 = width//4
    y0 = height//4
    new_width = width//2
    new_height = height//2
    center = frame[y0:y0+new_height, x0:x0+new_width].copy()
    return center


def video_capture(args):

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
    model = modelB(num_classes, True).to(device)
    model_name = type(model).__name__
    target_size = (224, 224)

    # loading snapshot
    snapshot_file = os.path.join("snapshots", model_name, "snapshot_%d.pt" % args.snapshot)
    if not os.path.exists(snapshot_file):
        print("Snapshot file does not exists: " + snapshot_file)
        return
    model.load_state_dict(torch.load(snapshot_file, 'cpu'))

    print("start video capture...")
    cap = cv2.VideoCapture(args.camera)

    # Check if camera opened successfully
    if not cap.isOpened():
        print("Error opening video stream or file")
        return

    font = cv2.FONT_HERSHEY_SIMPLEX

    past_frames = []

    while cap.isOpened():
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret:
            past_frames.append(clip(frame))
            if len(past_frames) > max(1, args.rtta):
                past_frames.pop(0)

            frame_to_show = cv2.resize(past_frames[-1], (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_AREA)

            # getting required number of frames for prediction
            images_to_predict = []
            if args.rtta < 1:
                images_to_predict.append(past_frames[-1])
            else:
                images_to_predict = past_frames[0: min(args.rtta, len(past_frames))]

            probs = predict_with_tta(args.tta, images_to_predict, model, target_size)
            sorted_indexes = np.argsort(probs)
            for t in range(5):
                label_index = sorted_indexes[-t - 1]
                label_prob = probs[label_index]
                label_name = labels[label_index]
                s = "%1.2f%% %s" % (label_prob * 100.0, label_name)
                cv2.putText(frame_to_show, s, (0, (t+1)*32), font, 1, (0,0,180), 2, cv2.LINE_AA)

            # Display the resulting frame
            cv2.imshow('Frame', frame_to_show)

            # Press Q on keyboard to  exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        # Break the loop
        else:
            break

    # When everything done, release the video capture object
    cap.release()

    # Closes all the frames
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run prediction from web camera')
    parser.add_argument("model", type=str, default=None, help="Name of the model to train.")
    parser.add_argument("--data_root", type=str, default=None, help="Path to data.")
    parser.add_argument("--snapshot", type=int, default=None, help="Iteration of a snapshot.")
    parser.add_argument("--gpu", type=str, default='true', help="Set to 'true' for GPU.")
    parser.add_argument("--camera", type=int, default=0, help="camera device index")
    parser.add_argument("--tta", type=int, default=0, help="0 - no TTA, 1 - hflip, 2 - vflip+hflip")
    parser.add_argument("--rtta", type=int, default=0,
                        help="Robot TTA. <1 - no TTA, >1 - number of images to take for TTA")
    parser.add_argument("--tta_mode", type=str, default="mean", help="'mean' or 'majority' voting TTA")

    _args = parser.parse_args()
    video_capture(_args)
