import math
import os

import numpy as np
import cv2
import torch
import torch.nn as nn
import argparse

from torch.utils.data import DataLoader

from DebugDumper import DebugDumper
from ImageDataset import ImageDataset
from PlotUtils import plot_histograms, plot_train_curves
from RenderingDataset import RenderingDataset
from TestDataset import TestDataset


def train(data_loader: DataLoader, model: nn.Module, num_iterations: int, start_iteration: int, device: torch.device,
          losses_dict: dict, accuracies_dict: dict, optimizer: torch.optim.Optimizer = None,
          debug_dumper: DebugDumper = None, log_file_name: str = None):

    loss_func = nn.CrossEntropyLoss()

    iterator = iter(data_loader)
    epoch_size = len(data_loader.dataset)

    is_train = optimizer is not None
    if is_train:
        model.train()
    else:
        model.eval()

    global_av_loss = 0
    global_av_accuracy = 0
    for i in range(num_iterations):

        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(data_loader)
            batch = next(iterator)

        # debug dump
        if debug_dumper is not None:
            debug_dumper.dump(batch)

        # transferring data to device
        for k in batch:
            batch[k] = batch[k].to(device)

        av_loss = 0
        av_accuracy = 0
        av_n = 0
        with torch.set_grad_enabled(is_train):

            predictions = model.forward(batch)

            labels = batch['label']
            labels = labels.view(labels.size(0))
            loss = loss_func(predictions, labels)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            av_loss += loss.item()
            global_av_loss += loss.item()

            # calculating accuracy
            predicted_labels = torch.argmax(predictions, dim=1)
            correct_count = predicted_labels.eq(labels).sum().item()
            accuracy = correct_count / labels.size(0)
            av_accuracy += accuracy
            global_av_accuracy += accuracy

            av_n += 1

        iteration = i + start_iteration
        if i % 10 == 0:
            e = iteration / epoch_size
            av_loss /= av_n
            av_accuracy = 100 * av_accuracy / av_n
            if is_train:
                log_string = "%d: epoch=%1.2f, loss=%f, accuracy=%1.2f%%" % (iteration, e, av_loss, av_accuracy)
            else:
                log_string = "%d: loss=%f, accuracy=%1.2f%%" % (i, av_loss, av_accuracy)
            print(log_string)
            if log_file_name is not None:
                with open(log_file_name, 'a+') as lf:
                    lf.write(log_string + '\n')

        # adding loss to log
        if is_train:
            losses_dict[iteration] = loss.item()
            accuracies_dict[iteration] = accuracy * 100.0

    if not is_train:
        global_av_loss /= num_iterations
        global_av_accuracy /= num_iterations
        print("average test loss = %f, average test accuracy = %f" % (global_av_loss, global_av_accuracy))
        losses_dict[num_iterations + start_iteration] = global_av_loss
        accuracies_dict[num_iterations + start_iteration] = global_av_accuracy

    return num_iterations + start_iteration


def image_to_tensor(img: np.ndarray) -> torch.Tensor:
    """
    Converts 8 bit BGR image [H, W, C] to 32-bit float tensor [1, C, H, W]
    :param img: 8 bit BGR image to convert.
    :return: torch tensor [1, C, H, W] float 32 [0..1]
    """
    img = torch.from_numpy(img)
    img = img.permute(2, 0, 1)
    img = img.unsqueeze(0)
    img = img.to(torch.float32) / 255.
    return img


def test(test_set: TestDataset, model: nn.Module, iteration: int, device: torch.device, accuracies_dict: dict, log_file_name: str):
    """
    Runs prediction on test set and calculates test accuracy.
    :return:
    """

    count = 0
    number = 0

    with torch.no_grad():
        model.eval()

        # iterating over all test set
        batch_size = 32
        num_batches = math.ceil(len(test_set) / float(batch_size))
        for b in range(num_batches):

            # assembling batch
            img_batch = []
            label_batch = []
            for i in range(batch_size):
                index = i + b * batch_size
                if index >= len(test_set):
                    break
                img, label = test_set[index]
                img_data = image_to_tensor(img)
                img_batch.append(img_data)
                label_batch.append(label)

            img_batch = torch.cat(img_batch, dim=0).to(device)
            label_batch = torch.tensor(label_batch, dtype=torch.long).to(device)

            probs = model.forward({'rgb': img_batch})
            # [B, num_classes]

            top_1 = torch.argmax(probs, dim=1)
            # [B]
            acc = torch.nonzero(top_1 == label_batch, as_tuple=False).shape[0]
            count += acc
            number += label_batch.shape[0]

    accuracy = 100.0 * count / number if number != 0 else 0.0
    accuracies_dict[iteration] = accuracy

    with open(log_file_name, 'a+') as f:
        f.write('%d, accuracy=%1.2f\n' % (iteration, accuracy))

    print('test accuracy = %1.2f%%' % accuracy)

    return accuracy


def set_num_threads(nt):
    try:
        import mkl
        mkl.set_num_threads(nt)
    except Exception as e:
        print('Unable to set numthreads in mkl: ' + str(e))

    cv2.setNumThreads(nt)

    nt = str(nt)
    os.environ['OPENBLAS_NUM_THREADS'] = nt
    os.environ['NUMEXPR_NUM_THREADS'] = nt
    os.environ['OMP_NUM_THREADS'] = nt
    os.environ['MKL_NUM_THREADS'] = nt


def main(args):

    batch_size = 32
    snapshot_iters = 500
    test_iters = 100
    snapshot_dir = "snapshots"
    image_size = (224, 224)
    learning_rate = 0.01
    momentum = 0.9
    debug_dir = "debug"
    debug_number = 0
    freeze_encoder = args.freeze_encoder.lower() == 'true'

    set_num_threads(1)

    use_cuda = torch.cuda.is_available() and args.gpu.lower() == 'true'
    device = torch.device("cuda" if use_cuda else "cpu")
    print("Using GPU" if use_cuda else "Using CPU")

    # create data sets
    train_foregrounds = ImageDataset(os.path.join(args.data_root, "train.txt"), "sorted")
    val_foregrounds = ImageDataset(os.path.join(args.data_root, "val.txt"), "sorted")
    num_classes = len(train_foregrounds.labels)
    backgrounds = ImageDataset(os.path.join(args.data_root, "backgrounds"), force_rgb=True)

    # initialize test set
    test_set = TestDataset(args.data_root, "test", dst_img_size=image_size)

    print("Number of classes = %d" % num_classes)
    print("Number of train foregrounds = %d" % len(train_foregrounds))
    print("Number of validation foregrounds = %d" % len(val_foregrounds))
    print("Number of test images = %d" % len(test_set))
    print("Number of backgrounds = %d" % len(backgrounds))

    if len(train_foregrounds) == 0 or len(val_foregrounds) == 0 or len(backgrounds) == 0:
        raise Exception("One of datasets is empty")

    train_dataset = RenderingDataset(backgrounds, train_foregrounds, image_size)
    test_dataset = RenderingDataset(backgrounds, val_foregrounds, image_size)

    # create data loaders
    kwargs = {'num_workers': 6, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
    val_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, **kwargs)

    # creating model
    model, model_name = load_model(args.model, device, num_classes, inference=False, freeze_encoder=freeze_encoder)

    # preparing snapshot dir
    if not os.path.exists(snapshot_dir):
        os.mkdir(snapshot_dir)

    # preparing model dir
    model_dir = os.path.join(snapshot_dir, model_name)
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    iteration = 0
    if args.snapshot is not None:
        iteration = args.snapshot
        snapshot_file = get_snapshot_file_name(iteration, model_dir)
        print("loading " + snapshot_file)
        model.load_state_dict(torch.load(snapshot_file))

    print('Starting from iteration %d' % iteration)

    # creating optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

    # creating logs
    train_losses = {}
    train_accuracies = {}
    val_losses = {}
    val_accuracies = {}
    test_accuracies = {}

    # creating debug dumper
    debug_dumper = DebugDumper(debug_dir, debug_number)

    # creating log files
    train_log_file = os.path.join(model_dir, 'train.log')
    val_log_file = os.path.join(model_dir, 'val.log')
    test_log_file = os.path.join(model_dir, 'test.log')

    while iteration < args.num_iterations:

        # train
        print("training...")
        iteration = train(train_loader, model, snapshot_iters, iteration, device, train_losses, train_accuracies,
                          optimizer, debug_dumper, train_log_file)

        # dumping snapshot
        snapshot_file = get_snapshot_file_name(iteration, model_dir)
        print("dumping snapshot: " + snapshot_file)
        torch.save(model.state_dict(), snapshot_file)

        # validation
        print("validation...")
        train(val_loader, model, test_iters, iteration, device, val_losses, val_accuracies,
              log_file_name=val_log_file)

        # test
        print("test...")
        test(test_set, model, iteration, device, test_accuracies, test_log_file)

        # visualizing training progress
        plot_histograms(model, model_dir)
        plot_train_curves(train_losses, val_losses, "TrainCurves", model_dir)
        plot_train_curves(train_accuracies, val_accuracies, "Validation Accuracy", model_dir, logy=False)
        plot_train_curves(val_accuracies, test_accuracies, "Test Accuracy", model_dir, logy=False)


def load_model(name, device, num_classes, inference, **kwargs):
    model_name = 'model_' + name
    module = __import__(model_name)
    _class = getattr(module, model_name)
    model = _class(num_classes, inference=inference, **kwargs).to(device)
    return model, model_name


def get_snapshot_file_name(iteration, model_dir):
    snapshot_file = os.path.join(model_dir, "snapshot_" + str(iteration) + ".pt")
    return snapshot_file


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run training and test after every N iterations.')
    parser.add_argument("model", type=str, default=None, help="Name of the model to train.")
    parser.add_argument("--data_root", type=str, default=None, help="Path to data.")
    parser.add_argument("--snapshot", type=int, default=None, help="Iteration to start from snapshot.")
    parser.add_argument("--num_iterations", type=int, default=100000, help="Number of iterations to train.")
    parser.add_argument("--gpu", type=str, default='true', help="Set to 'true' for GPU.")
    parser.add_argument("--freeze_encoder", type=str, default='true', help="Set to 'true' to freeze encoder layers.")
    _args = parser.parse_args()
    main(_args)
