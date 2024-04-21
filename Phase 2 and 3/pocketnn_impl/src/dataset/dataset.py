import numpy as np
from typing import List, Tuple
from typing import Union, Dict
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import InnerDirichletPartitioner

DatasetTuple = Tuple[np.ndarray, np.ndarray]


def get_dataset(dataset_name: str) -> Tuple[DatasetTuple, DatasetTuple]:
    assert dataset_name in ['mnist', 'fashion_mnist']

    if dataset_name.lower() == 'mnist':
        dataset = FederatedDataset(dataset="mnist", partitioners={})  # No partitioning
    elif dataset_name.lower() == 'fashion_mnist':
        dataset = FederatedDataset(dataset="fashion_mnist", partitioners={})  # No partitioning
    else:
        raise ValueError('Invalid dataset name')

    train_dataset = dataset.load_split('train').with_format('numpy')
    train_images, train_labels = train_dataset['image'].reshape(-1, 28 * 28), train_dataset['label']

    test_dataset = dataset.load_split('test').with_format('numpy')
    test_images, test_labels = test_dataset['image'].reshape(-1, 28 * 28), test_dataset['label']

    train_data = (train_images, train_labels)
    test_data = (test_images, test_labels)

    return train_data, test_data


ClientDataset = Dict[str, Union[None, DatasetTuple]]


def load_dataset(dataset_name: str, dirichlet_alpha: Union[int, float], num_clients: int, train_ratio: float) \
        -> Tuple[List[ClientDataset], DatasetTuple]:
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

    client_dataset = []
    for c in range(num_clients):
        partition = fds.load_partition(c).with_format('numpy')
        if not np.isclose(train_ratio, 1.0):  # split into train and test if train_ratio is not 1
            partition = partition.train_test_split(
                train_size=int(each_client_data * train_ratio))  # ratio 1 is not supported

            train_image, train_label = partition['train']['image'], partition['train']['label']
            train_image = train_image.reshape(-1, 28 * 28)
            client_train_dataset = (train_image, train_label)

            test_image, test_label = partition['test']['image'], partition['test']['label']
            test_image = test_image.reshape(-1, 28 * 28)
            client_test_dataset = (test_image, test_label)
        else:
            train_image, train_label = partition['image'], partition['label']
            train_image = train_image.reshape(-1, 28 * 28)
            client_train_dataset = (train_image, train_label)
            client_test_dataset = None

        client_dataset.append({
            'train': client_train_dataset,
            'test': client_test_dataset
        })

    test_partition = fds.load_split('test').with_format('numpy')
    test_image, test_label = test_partition['image'], test_partition['label']
    test_image = test_image.reshape(-1, 28 * 28)
    test_dataset = (test_image, test_label)

    return client_dataset, test_dataset
