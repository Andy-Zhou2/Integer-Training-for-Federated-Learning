from typing import List, Tuple
from typing import Union, Dict
from .mix_dataset import DatasetTuple, get_centralized_dataloader, load_federated_dataset


def get_centralized_dataloader_pkt(dataset_name: str) -> Tuple[DatasetTuple, DatasetTuple]:
    assert dataset_name in ['mnist', 'fashion_mnist']

    return get_centralized_dataloader(dataset_name, post_process='pktnn')


ClientDataset = Dict[str, Union[None, DatasetTuple]]


def load_federated_dataset_pkt(dataset_name: str, dirichlet_alpha: Union[int, float], num_clients: int, train_ratio: float) \
        -> Tuple[List[ClientDataset], DatasetTuple]:
    assert dataset_name in ['mnist', 'fashion_mnist']

    return load_federated_dataset(dataset_name, dirichlet_alpha, num_clients, train_ratio, shuffle=True,
                                  post_process='pktnn')
