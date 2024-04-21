from torchvision import transforms
from torch.utils.data import DataLoader
from typing import Tuple, List, Union, Dict
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import InnerDirichletPartitioner
import numpy as np

DatasetTuple = Tuple[np.ndarray, np.ndarray]


def apply_transforms(batch):
    """standard transform for fp dataset"""
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

    batch["image"] = [transform(img) for img in batch["image"]]
    return batch


DataPartition = Union[DataLoader, DatasetTuple]  # DataLoader for fp or DatasetTuple for pktnn


def get_centralized_dataloader(dataset_name: str, post_process: str, batch_size: int = 0) -> \
        Tuple[DataPartition, DataPartition]:
    assert post_process in ['pktnn', 'fp']
    if post_process == 'pktnn':
        assert batch_size == 0, 'batch_size is not a valid argument for pktnn'

    if dataset_name.lower() == 'mnist':
        dataset = FederatedDataset(dataset="mnist", partitioners={})  # No partitioning
    elif dataset_name.lower() == 'fashion_mnist':
        dataset = FederatedDataset(dataset="fashion_mnist", partitioners={})  # No partitioning
    else:
        raise ValueError('Invalid dataset name')

    if post_process == 'pktnn':
        train_dataset = dataset.load_split('train').with_format('numpy')
        train_images, train_labels = train_dataset['image'].reshape(-1, 28 * 28), train_dataset['label']

        test_dataset = dataset.load_split('test').with_format('numpy')
        test_images, test_labels = test_dataset['image'].reshape(-1, 28 * 28), test_dataset['label']

        train_data = (train_images, train_labels)
        test_data = (test_images, test_labels)

        return train_data, test_data
    else:
        full_dataset = dataset.load_split('train').with_transform(apply_transforms)
        train_loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=True)

        test_dataset = dataset.load_split('test').with_transform(apply_transforms)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


ClientDataset = Dict[str, Union[None, DataPartition]]


def load_federated_dataset(dataset_name: str, dirichlet_alpha: Union[int, float], num_clients: int, train_ratio: float,
                           shuffle: bool, post_process: str, batch_size: int = 0) \
        -> Tuple[List[ClientDataset], DataPartition]:
    assert dataset_name in ['mnist', 'fashion_mnist']
    assert post_process in ['pktnn', 'fp']
    if post_process == 'pktnn':
        assert batch_size == 0, 'batch_size is not a valid argument for pktnn'

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
        if post_process == 'pktnn':
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
        else:  # post_process == 'fp'
            partition = fds.load_partition(c).with_transform(apply_transforms)
            if not np.isclose(train_ratio, 1.0):  # split into train and test if train_ratio is not 1
                partition = partition.train_test_split(
                    train_size=int(each_client_data * train_ratio))  # ratio 1 is not supported

                client_train_dataset = DataLoader(partition['train'], batch_size=batch_size, shuffle=shuffle)
                client_test_dataset = DataLoader(partition['test'], batch_size=batch_size, shuffle=False)
            else:
                client_train_dataset = DataLoader(partition, batch_size=batch_size, shuffle=shuffle)
                client_test_dataset = None

        client_dataset.append({
            'train': client_train_dataset,
            'test': client_test_dataset
        })

    if post_process == 'pktnn':
        test_partition = fds.load_split('test').with_format('numpy')
        test_image, test_label = test_partition['image'], test_partition['label']
        test_image = test_image.reshape(-1, 28 * 28)
        test_dataset = (test_image, test_label)
    else:
        test_partition = fds.load_split('test').with_transform(apply_transforms)
        test_dataset = DataLoader(test_partition, batch_size=batch_size, shuffle=False)

    return client_dataset, test_dataset
