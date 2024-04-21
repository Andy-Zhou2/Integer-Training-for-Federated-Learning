from torchvision import transforms
from torch.utils.data import DataLoader
from typing import Tuple, List, Union, Dict
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import InnerDirichletPartitioner
import numpy as np
from .mix_dataset import get_centralized_dataloader, load_federated_dataset


def get_centralized_dataloader_fp(dataset_name: str, batch_size: int) -> Tuple[DataLoader, DataLoader]:
    assert dataset_name in ['mnist', 'fashion_mnist']

    return get_centralized_dataloader(dataset_name, post_process='fp', batch_size=batch_size)


ClientDataset = Dict[str, Union[None, DataLoader]]


def load_federated_dataset_fp(dataset_name: str, dirichlet_alpha: Union[int, float], num_clients: int,
                              train_ratio: float,
                              batch_size: int, shuffle: bool) -> Tuple[List[ClientDataset], DataLoader]:
    assert dataset_name in ['mnist', 'fashion_mnist']

    return load_federated_dataset(dataset_name, dirichlet_alpha, num_clients, train_ratio, shuffle, post_process='fp',
                                  batch_size=batch_size)
