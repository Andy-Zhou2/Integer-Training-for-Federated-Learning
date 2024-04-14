import os
from network import get_net
from train_evaluate import pktnn_train, pktnn_evaluate
from dataset import get_dataset

print('Loading data')

dataset_name = 'mnist'  # mnist or fashion_mnist
assert dataset_name in ['mnist', 'fashion_mnist']

train_data, test_data = get_dataset(dataset_name)

# create folder
weight_folder = f'../data/weights/{dataset_name}'
os.makedirs(weight_folder, exist_ok=True)

print('Creating model')
net = get_net(dataset_name + '_default')

# initial testing
print('Initial testing')
acc = pktnn_evaluate(net, train_data)
print(f'Initial training accuracy: {acc * 100}%')
acc = pktnn_evaluate(net, test_data)
print(f'Initial testing accuracy: {acc * 100}%')

config = {
    'epochs': 1,
    'batch_size': 20,
    'initial_lr_inv': 1000,
    'weight_folder': weight_folder,
    'test_every_epoch': True,
    'print_hash_every_epoch': True,
    'shuffle_dataset_every_epoch': False,
    'verbose': True
}
pkt_data = {
    'train': train_data,
    'test': test_data
}
pktnn_train(net, pkt_data, config)

