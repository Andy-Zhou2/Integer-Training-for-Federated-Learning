import numpy as np
from torchvision import datasets, transforms
from pktnn_fc import PktFc
from pktnn_mat import PktMat
from pktnn_consts import UNSIGNED_4BIT_MAX
from pktnn_loss import batch_l2_loss_delta
from state import save_state, load_state
import os
from network import MNISTNet, FashionMNISTNet
from train import pktnn_train, pktnn_evaluate

print('Loading data')

dataset_name = 'fashion_mnist'  # mnist or fashion_mnist
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

# create folder
weight_folder = f'../data/weights/{dataset_name}'
os.makedirs(weight_folder, exist_ok=True)

num_train_samples = len(dataset_train)
num_test_samples = len(dataset_test)

print('Transforming data')

train_images = dataset_train.data.numpy().reshape(-1, 28 * 28)
train_labels = dataset_train.targets.numpy()
test_images = dataset_test.data.numpy().reshape(-1, 28 * 28)
test_labels = dataset_test.targets.numpy()

train_data = (train_images, train_labels)
test_data = (test_images, test_labels)


print('Creating model')
if dataset_name == 'mnist':
    net = MNISTNet()
elif dataset_name == 'fashion_mnist':
    net = FashionMNISTNet()
else:  # should not reach here
    raise ValueError('Invalid dataset name')

# initial testing
print('Initial testing')
acc = pktnn_evaluate(net, train_data)
print(f'Initial training accuracy: {acc * 100}%')
acc = pktnn_evaluate(net, test_data)
print(f'Initial testing accuracy: {acc * 100}%')

config = {
    'epochs': 100,
    'batch_size': 20,
    'initial_lr_inv': 1000,
    'weight_folder': weight_folder,
    'test_every_epoch': True,
    'print_hash_every_epoch': True
}
pkt_data = {
    'train': train_data,
    'test': test_data
}
pktnn_train(net, pkt_data, config)

