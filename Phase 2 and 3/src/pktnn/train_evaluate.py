import numpy as np
from .pkt_network import PktNet
from typing import Dict, Any, Tuple, List
from .pkt_mat import PktMat
from .pkt_loss import batch_l2_loss_delta
import os
import logging


def pktnn_train(net: PktNet, data: Dict[str, Tuple[np.ndarray, np.ndarray]], config: Dict[str, Any]) \
        -> Dict[str, List[float]]:
    """
    Train the network on the given images and labels. Modify the network in place.

    :param net: The network to train
    :param data: A dictionary containing train / test dataset, where each dataset is a tuple of images and labels
    :param config: A dictionary containing training configuration
    :return: None
    """
    assert 'train' in data, "Train dataset is required"
    train_data = data['train']
    train_images: np.ndarray = train_data[0]
    train_labels: np.ndarray = train_data[1]
    assert train_images.shape[0] == train_labels.shape[0], "Number of images and labels should be the same"
    num_train_samples = train_images.shape[0]

    result = {
        'loss': [],
        'train_accuracy': [],
        'test_accuracy': [],
    }

    num_classes = 10
    mnist_rows = 28
    mnist_cols = 28
    dim_input = mnist_rows * mnist_cols

    mnist_train_images = PktMat(num_train_samples, dim_input)
    mnist_train_images.mat = train_images.reshape(-1, dim_input)

    train_target_mat = PktMat(num_train_samples, num_classes)
    for r in range(num_train_samples):
        train_target_mat[r][train_labels[r]] = config['label_target_value']

    EPOCH = config['epochs']
    BATCH_SIZE = config['batch_size']  # too big could cause overflow

    lr_inv = np.int_(config['initial_lr_inv'])

    indices = np.arange(num_train_samples)

    for epoch in range(1, EPOCH + 1):
        if config['verbose']:
            logging.info(f'Epoch {epoch}')

        if config['shuffle_dataset_every_epoch']:
            np.random.shuffle(indices)

        if epoch % 10 == 0 and lr_inv < 2 * lr_inv:
            # avoid overflow
            lr_inv *= 2

        sum_loss = 0
        epoch_num_correct = 0
        num_iter = num_train_samples // BATCH_SIZE

        for i in range(num_iter):
            mini_batch_images = PktMat(BATCH_SIZE, dim_input)
            mini_batch_train_targets = PktMat(BATCH_SIZE, num_classes)

            idx_start = i * BATCH_SIZE
            idx_end = idx_start + BATCH_SIZE
            mini_batch_images[0:BATCH_SIZE] = mnist_train_images[indices[idx_start:idx_end]]
            mini_batch_train_targets[0:BATCH_SIZE] = train_target_mat[indices[idx_start:idx_end]]

            output = net.forward(mini_batch_images)

            sum_loss += sum(np.square(mini_batch_train_targets.mat - output.mat).flatten() // 2)
            loss_delta_mat = batch_l2_loss_delta(mini_batch_train_targets, output)

            for r in range(BATCH_SIZE):
                if mini_batch_train_targets.get_max_index_in_row(r) == output.get_max_index_in_row(r):
                    epoch_num_correct += 1

            net.backward(loss_delta_mat, lr_inv)

        train_acc = epoch_num_correct / num_train_samples
        result['loss'].append(sum_loss)
        result['train_accuracy'].append(train_acc)

        if config['print_hash_every_epoch']:
            fc_list = net.get_fc_list()
            params_sum = sum([fc.weight.sum() + fc.bias.sum() for fc in fc_list])
            logging.info(f'epoch {epoch}: has param sum {params_sum}')
            for fc in fc_list:
                logging.info(f'fc hash: {fc.weight.hash()} {fc.bias.hash()}')

        # save state
        weight_folder = config.get('weight_folder', '')
        if weight_folder:
            os.makedirs(weight_folder, exist_ok=True)
            net.save(os.path.join(weight_folder, f'epoch_{epoch}.npz'))

        if config['verbose']:
            logging.info(f'Epoch {epoch}, loss: {sum_loss}, accuracy: {train_acc * 100}%')

            # for each fc layer, print the 10%, 20%, ...100% percentile of the weights and biases
            for fc in net.get_fc_list():
                weight_percentiles = np.percentile(fc.weight.mat, np.arange(0, 101, 10))
                bias_percentiles = np.percentile(fc.bias.mat, np.arange(0, 101, 10))
                print(f'FC Layer: {fc.in_dim} -> {fc.out_dim}')
                print(f'Weight percentiles: {weight_percentiles}')
                print(f'Bias percentiles: {bias_percentiles}')


            # for each fc layer, print the 10%, 20%, ...100% percentile of the weights and biases
            for fc in net.get_fc_list():
                weight_percentiles = np.percentile(fc.weight.mat, np.arange(0, 101, 10))
                bias_percentiles = np.percentile(fc.bias.mat, np.arange(0, 101, 10))
                print(f'FC Layer: {fc.in_dim} -> {fc.out_dim}')
                print(f'Weight percentiles: {weight_percentiles}')
                print(f'Bias percentiles: {bias_percentiles}')


        if config['test_every_epoch']:
            assert 'test' in data, "Test dataset is required if test_every_epoch is True"
            test_data = data['test']
            acc = pktnn_evaluate(net, test_data)
            result['test_accuracy'].append(acc)
            logging.info(f"Epoch {epoch}, testing accuracy: {acc * 100}%")
    return result


def pktnn_evaluate(net: PktNet, test_data: Tuple[np.ndarray, np.ndarray]) -> float:
    """
    Evaluate the network on the given test dataset.

    :param net: The network to evaluate
    :param test_data: A tuple containing test images and labels
    :return: The accuracy of the network on the test dataset
    """
    test_images: np.ndarray = test_data[0]
    test_labels: np.ndarray = test_data[1]
    assert test_images.shape[0] == test_labels.shape[0], "Number of images and labels should be the same"
    num_test_samples = test_images.shape[0]

    mnist_rows = 28
    mnist_cols = 28
    dim_input = mnist_rows * mnist_cols

    mnist_test_images = PktMat(num_test_samples, dim_input)
    mnist_test_images.mat = test_images.reshape(num_test_samples, dim_input)

    output = net.forward(mnist_test_images)
    num_correct = 0
    for r in range(num_test_samples):
        if test_labels[r] == output.get_max_index_in_row(r):
            num_correct += 1

    return num_correct / num_test_samples
