import numpy as np
from torchvision import datasets, transforms
from typing import List, Tuple

# typing for dataset tuple
DatasetTuple = Tuple[np.ndarray, np.ndarray]


def get_dataset(dataset_name: str) -> Tuple[DatasetTuple, DatasetTuple]:
    assert dataset_name in ['mnist', 'fashion_mnist']

    if dataset_name == 'mnist':
        dataset_train = datasets.MNIST('../data', train=True,
                                       download=True)
        dataset_test = datasets.MNIST('../data', train=False,
                                      download=True)
    elif dataset_name == 'fashion_mnist':
        dataset_train = datasets.FashionMNIST('../data', train=True,
                                              download=True)
        dataset_test = datasets.FashionMNIST('../data', train=False,
                                             download=True)
    else:  # should not reach here
        raise ValueError('Invalid dataset name')

    train_images = dataset_train.data.numpy().reshape(-1, 28 * 28)
    train_labels = dataset_train.targets.numpy()
    test_images = dataset_test.data.numpy().reshape(-1, 28 * 28)
    test_labels = dataset_test.targets.numpy()

    train_data = (train_images, train_labels)
    test_data = (test_images, test_labels)

    return train_data, test_data
