import numpy as np
from torchvision import datasets, transforms
from pktnn_fc import PktFc
from pktnn_mat import PktMat
from pktnn_consts import UNSIGNED_4BIT_MAX
from pktnn_loss import batch_l2_loss_delta
from typing import List, Tuple
import random
from collections import OrderedDict, defaultdict
import numbers

import flwr as fl
from flwr.common import Metrics
from flwr_datasets import FederatedDataset
from mnist_net import MNISTNet

NUM_CLIENTS = 5


def load_datasets():
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    dataset_train = datasets.MNIST('./data', train=True,
                                   transform=transform, download=True)
    dataset_test = datasets.MNIST('./data', train=False,
                                  transform=transform, download=True)

    dataset_train = [((data * 256).numpy().astype(np.int_), answer) for data, answer in dataset_train]
    dataset_test = [((data * 256).numpy().astype(np.int_), answer) for data, answer in dataset_test]

    # split the training set into NUM_CLIENTS partitions, and each partition into train and validation sets
    random.shuffle(dataset_train)

    size_each_client = len(dataset_train) // NUM_CLIENTS
    size_training_each_client = int(size_each_client * 0.8)

    train_datasets = []
    val_datasets = []
    for partition_id in range(NUM_CLIENTS):
        start = partition_id * size_each_client
        end = start + size_each_client
        train_datasets.append(dataset_train[start:start + size_training_each_client])
        val_datasets.append(dataset_train[start + size_training_each_client:end])
    return train_datasets, val_datasets, dataset_test


train_loaders, val_loaders, test_loader = load_datasets()


def train(net, dataset_train, epochs: int, config: dict):
    num_train_samples = len(dataset_train)

    num_classes = 10
    mnist_rows = 28
    mnist_cols = 28

    dim_input = mnist_rows * mnist_cols

    mnist_train_labels = PktMat(num_train_samples, 1)
    mnist_train_images = PktMat(num_train_samples, dim_input)
    for i in range(num_train_samples):
        mnist_train_labels[i][0] = dataset_train[i][1]
        mnist_train_images[i] = dataset_train[i][0].flatten()

    fc1, fc2, fc_last = net.fc1, net.fc2, net.fc_last

    train_target_mat = PktMat(num_train_samples, num_classes)
    for r in range(num_train_samples):
        train_target_mat[r][mnist_train_labels[r][0]] = UNSIGNED_4BIT_MAX

    loss_delta_mat = PktMat()

    EPOCH = 1
    BATCH_SIZE = 20  # too big could cause overflow

    lr_inv = np.int_(1000)

    indices = np.arange(num_train_samples)

    for epoch in range(EPOCH):
        # shuffle indices
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
            # print(f'mini_batch_images: {mini_batch_images}')
            # print(f'mini_batch_train_targets: {mini_batch_train_targets.mat}')

            fc1.forward(mini_batch_images)

            sum_loss += sum(1 / 2 * np.square(mini_batch_train_targets.mat - fc_last.output.mat).flatten())
            sum_loss_delta = batch_l2_loss_delta(loss_delta_mat, mini_batch_train_targets, fc_last.output)

            for r in range(BATCH_SIZE):
                # print(f'for sample {r}, max index in target: {mini_batch_train_targets.get_max_index_in_row(r)},\
                #  max index in output: {fc_last.output.get_max_index_in_row(r)}')

                if mini_batch_train_targets.get_max_index_in_row(r) == fc_last.output.get_max_index_in_row(r):
                    epoch_num_correct += 1

            fc_last.backward(loss_delta_mat, lr_inv)
        params = get_parameters(net)
        print(f'epoch {epoch}: {config["cid"]} has param sum {sum([np.sum(p) for p in params])}')
        # print(f'cid {config["cid"]} Round {config["server_round"]} Epoch {epoch}, loss: {sum_loss / num_train_samples}, accuracy: {epoch_num_correct / num_train_samples * 100}%')


def test(net, dataset_test, config):
    num_test_samples = len(dataset_test)
    mnist_rows = 28
    mnist_cols = 28
    dim_input = mnist_rows * mnist_cols
    mnist_test_images = PktMat(num_test_samples, dim_input)
    for i in range(num_test_samples):
        mnist_test_images[i] = dataset_test[i][0].flatten()
    fc1, fc2, fc_last = net.fc1, net.fc2, net.fc_last

    num_correct = 0

    fc1.forward(mnist_test_images)
    for r in range(num_test_samples):
        if fc_last.output.get_max_index_in_row(r) == dataset_test[r][1]:
            num_correct += 1

    accuracy = num_correct / num_test_samples
    # print(f'cid {config["cid"]} Round {config["server_round"]} Testing accuracy: {accuracy * 100}%')
    return float('nan'), accuracy  # loss, accuracy


def set_parameters(net, parameters: List[np.ndarray]):
    # convert each parameter to integer
    for i in range(len(parameters)):
        parameters[i] = parameters[i].astype(np.int_)
    # print(f'setting parameters {parameters}')

    fc1, fc2, fc_last = net.fc1, net.fc2, net.fc_last
    fc1.weight.mat = parameters[0]
    fc1.bias.mat = parameters[1]
    fc2.weight.mat = parameters[2]
    fc2.bias.mat = parameters[3]
    fc_last.weight.mat = parameters[4]
    fc_last.bias.mat = parameters[5]


def get_parameters(net) -> List[np.ndarray]:
    fc1, fc2, fc_last = net.fc1, net.fc2, net.fc_last
    return [fc1.weight.mat, fc1.bias.mat, fc2.weight.mat, fc2.bias.mat, fc_last.weight.mat, fc_last.bias.mat]


class FlowerClient(fl.client.NumPyClient):
    def __init__(self, net, trainloader, valloader, cid: str):
        self.net = net
        self.train_loader = trainloader
        self.val_loader = valloader
        self.cid = cid

    def get_parameters(self, config):
        return get_parameters(self.net)

    def fit(self, parameters, config):
        # print(f'cid {self.cid} fitting with config {config}')
        set_parameters(self.net, parameters)
        print(f'before： {self.cid} has param sum {sum([np.sum(p) for p in parameters])}')
        train(self.net, self.train_loader, epochs=1, config=config | {'cid': self.cid})
        params = get_parameters(self.net)
        print(f'after: {self.cid} has param sum {sum([np.sum(p) for p in params])}')
        return params, len(self.train_loader), {'cid': self.cid}

    def evaluate(self, parameters, config):
        # print(f'evaluating {self.cid} with config {config}')
        set_parameters(self.net, parameters)
        loss, accuracy = test(self.net, self.val_loader, config | {'cid': self.cid})
        return float(loss), len(self.val_loader), {"accuracy": float(accuracy), 'cid': self.cid}


def client_fn(cid: str):
    # Load model
    net = MNISTNet()
    train_loader = train_loaders[int(cid)]
    val_loader = val_loaders[int(cid)]

    return FlowerClient(net, train_loader, val_loader, cid).to_client()


client_resources = {"num_cpus": 1, "num_gpus": 0.0}


def aggregate_weighted_average(metrics: list[tuple[int, dict]]) -> dict:
    """Combine results from multiple clients.

    Args:
        metrics (list[tuple[int, dict]]): collected clients metrics

    Returns
    -------
        dict: result dictionary containing the aggregate of the metrics passed.
    """
    average_dict: dict = defaultdict(list)
    total_examples: int = 0
    for num_examples, metrics_dict in metrics:
        for key, val in metrics_dict.items():
            if isinstance(val, numbers.Number):
                average_dict[key].append((num_examples, val))
        total_examples += num_examples
    return {
        key: {
            "avg": float(
                sum([num_examples * m for num_examples, m in val])
                / float(total_examples)
            ),
            "all": val,
        }
        for key, val in average_dict.items()
    }


def _on_fit_config_fn(server_round: int):
    return {"server_round": server_round}


def _on_evaluate_config_fn(server_round: int):
    return {"server_round": server_round}


def federated_evaluation_function(server_round: int, parameters, fed_eval_config):
    """returns (loss, dict of results)"""
    net = MNISTNet()
    set_parameters(net, parameters)
    loss, accuracy = test(net, test_loader, fed_eval_config | {'cid': 'server', 'server_round': server_round})
    return loss, {"accuracy": accuracy}


strategy = fl.server.strategy.FedAvg(
    fraction_fit=1.0,
    fraction_evaluate=1,
    min_fit_clients=NUM_CLIENTS,
    min_evaluate_clients=NUM_CLIENTS,
    min_available_clients=NUM_CLIENTS,
    evaluate_metrics_aggregation_fn=aggregate_weighted_average,
    on_fit_config_fn=_on_fit_config_fn,
    on_evaluate_config_fn=_on_evaluate_config_fn,
    evaluate_fn=federated_evaluation_function
)

# Start simulation
hist = fl.simulation.start_simulation(
    client_fn=client_fn,
    num_clients=NUM_CLIENTS,
    config=fl.server.ServerConfig(num_rounds=10),
    strategy=strategy,
    client_resources=client_resources,
)

print()
