from torchvision import datasets, transforms
import torch
from torch.utils.data import DataLoader
from typing import Tuple, List, Union, Dict
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import InnerDirichletPartitioner
import numpy as np


def get_centralized_dataloader(dataset_name: str, batch_size: int) -> Tuple[DataLoader, DataLoader]:
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_dataset = datasets.MNIST('../data', train=True, download=True, transform=transform) \
        if dataset_name == 'MNIST' else datasets.FashionMNIST('../data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('../data', train=False, download=True, transform=transform) \
        if dataset_name == 'MNIST' else datasets.FashionMNIST('../data', train=False, download=True,
                                                              transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


ClientDataset = Dict[str, Union[None, DataLoader]]


def load_federated_dataset(dataset_name: str, dirichlet_alpha: Union[int, float], num_clients: int, train_ratio: float,
                           batch_size: int, shuffle: bool) -> Tuple[List[ClientDataset], DataLoader]:
    assert dataset_name in ['mnist', 'fashion_mnist']

    if dataset_name == 'mnist':
        each_client_data = 60_000 // num_clients
        partitioner = InnerDirichletPartitioner(
            partition_sizes=[each_client_data] * num_clients, partition_by="label", alpha=dirichlet_alpha
        )
        fds = FederatedDataset(dataset="mnist", partitioners={"train": partitioner})
    elif dataset_name == 'fashion_mnist':
        each_client_data = 60_000 // num_clients
        partitioner = InnerDirichletPartitioner(
            partition_sizes=[each_client_data] * num_clients, partition_by="label", alpha=dirichlet_alpha
        )
        fds = FederatedDataset(dataset="fashion_mnist", partitioners={"train": partitioner})
    else:  # should not reach here
        raise ValueError('Invalid dataset name')

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

    client_dataset = []
    for c in range(num_clients):
        partition = fds.load_partition(c).with_transform(transform)
        if not np.isclose(train_ratio, 1.0):  # split into train and test if train_ratio is not 1
            partition = partition.train_test_split(
                train_size=int(each_client_data * train_ratio))  # ratio 1 is not supported

            client_train_dataloader = DataLoader(partition['train'], batch_size=batch_size, shuffle=shuffle)
            client_test_dataloader = DataLoader(partition['test'], batch_size=batch_size, shuffle=False)
        else:
            client_train_dataloader = DataLoader(partition, batch_size=batch_size, shuffle=shuffle)
            client_test_dataloader = None

        client_dataset.append({
            'train': client_train_dataloader,
            'test': client_test_dataloader
        })

    test_partition = fds.load_split('test').with_transform(transform)
    test_dataset = DataLoader(test_partition, batch_size=batch_size, shuffle=False)

    return client_dataset, test_dataset
