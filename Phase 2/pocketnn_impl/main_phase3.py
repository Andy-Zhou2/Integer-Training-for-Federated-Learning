import numpy as np
from typing import List, Tuple, Dict, Any
import random
from collections import OrderedDict, defaultdict
import numbers

import flwr as fl
from flwr.common import Metrics
from flwr_datasets import FederatedDataset
from network import get_net
from dataset import get_dataset, DatasetTuple
from network import PktNet
from flwr.common.typing import NDArrays, NDArrayInt
from train_evaluate import pktnn_train, pktnn_evaluate
from strategy import FedAvgInt

num_clients = 5
dataset_name = 'mnist'
num_rounds = 10
client_resources = {"num_cpus": 1, "num_gpus": 0.0}
client_train_config = {
    'epochs': 10,
    'batch_size': 20,
    'initial_lr_inv': 1000,
    'weight_folder': '',  # empty string: don't save
    'test_every_epoch': False,
    'print_hash_every_epoch': False,
    'shuffle_dataset_every_epoch': False,
    'verbose': False
}

ClientDataset = Dict[str, DatasetTuple]


def load_datasets(dataset_name: str) -> Tuple[List[ClientDataset], DatasetTuple]:
    """
    Load the dataset and split the training set into NUM_CLIENTS partitions.

    :param dataset_name: the name of the dataset
    :return: a list of client datasets and the test dataset
    """
    train_data, test_data = get_dataset(dataset_name)
    (train_images, train_labels), (test_images, test_labels) = train_data, test_data

    # shuffle training dataset
    indices = np.arange(len(train_images))
    np.random.shuffle(indices)
    train_images = train_images[indices]
    train_labels = train_labels[indices]

    # split the training set into NUM_CLIENTS partitions, and each partition into train and validation sets
    size_each_client = len(train_images) // num_clients
    size_training_each_client = int(size_each_client * 0.8)

    client_dataset = []
    for partition_id in range(num_clients):
        start = partition_id * size_each_client
        end = start + size_each_client
        client_train_indices = indices[start:start + size_training_each_client]
        client_train_dataset = (train_images[client_train_indices], train_labels[client_train_indices])
        client_val_indices = indices[start + size_training_each_client:end]
        client_val_dataset = (train_images[client_val_indices], train_labels[client_val_indices])
        client_dataset.append({
            'train': client_train_dataset,
            'test': client_val_dataset
        })

    test_dataset = (test_images, test_labels)

    return client_dataset, test_dataset


client_datasets, test_dataset = load_datasets('mnist')


class FlowerClient(fl.client.NumPyClient):
    def __init__(self, net: PktNet, client_dataset: ClientDataset, cid: str):
        self.net = net
        self.client_dataset = client_dataset
        self.cid = cid

    def get_parameters(self, config):
        return self.net.get_parameters()

    def fit(self, parameters: NDArrays, config: Dict[str, Any]):
        self.net.set_parameters(parameters)
        # print(f'before: {self.cid} has param sum {sum([np.sum(p) for p in parameters])}')

        pktnn_train(self.net, self.client_dataset, config=config | {'cid': self.cid})
        params = self.net.get_parameters()
        # print(f'after: {self.cid} has param sum {sum([np.sum(p) for p in params])}')

        len_train_data = self.client_dataset['train'][0].shape[0]

        return params, len_train_data, {'cid': self.cid}

    def evaluate(self, parameters: NDArrays, config: Dict[str, Any]):
        self.net.set_parameters(parameters)
        accuracy = pktnn_evaluate(self.net, self.client_dataset['test'])
        len_val_data = self.client_dataset['test'][0].shape[0]

        return float('NAN'), len_val_data, {"accuracy": accuracy, 'cid': self.cid}


def client_fn(cid: str):
    # Load model
    net = get_net(dataset_name)
    cid_int = int(cid)
    client_dataset = client_datasets[cid_int]

    return FlowerClient(net, client_dataset, cid).to_client()


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
    return client_train_config | {"server_round": server_round}


def _on_evaluate_config_fn(server_round: int):
    return {"server_round": server_round}


def federated_evaluation_function(server_round: int, parameters: NDArrays, fed_eval_config):
    """returns (loss, dict of results)"""
    net = get_net(dataset_name)
    net.set_parameters(parameters)
    accuracy = pktnn_evaluate(net, test_dataset)
    return float('NAN'), {"accuracy": accuracy}


strategy = FedAvgInt(
    fraction_fit=1,
    fraction_evaluate=1,
    min_fit_clients=num_clients,
    min_evaluate_clients=num_clients,
    min_available_clients=num_clients,
    evaluate_metrics_aggregation_fn=aggregate_weighted_average,
    on_fit_config_fn=_on_fit_config_fn,
    on_evaluate_config_fn=_on_evaluate_config_fn,
    evaluate_fn=federated_evaluation_function
)

# Start simulation
hist = fl.simulation.start_simulation(
    client_fn=client_fn,
    num_clients=num_clients,
    config=fl.server.ServerConfig(num_rounds=num_rounds),
    strategy=strategy,
    client_resources=client_resources,
)
