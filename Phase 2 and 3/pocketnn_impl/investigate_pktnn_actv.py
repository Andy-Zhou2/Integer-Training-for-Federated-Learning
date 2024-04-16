import os
from network import get_net
from train_evaluate import pktnn_train, pktnn_evaluate
from dataset import get_dataset

print('Loading data')

dataset_name = 'mnist'  # mnist or fashion_mnist
assert dataset_name in ['mnist', 'fashion_mnist']

train_data, test_data = get_dataset(dataset_name)

# create folder
weight_folder = f'../data/weights/{dataset_name}-512'

print('Creating model')
net = get_net(dataset_name + '_default')
net.load(os.path.join(weight_folder, 'epoch_10.npz'))

# initial testing
print('Doing Inference...')
acc = pktnn_evaluate(net, train_data)
print(f'Initial training accuracy: {acc * 100}%')
acc = pktnn_evaluate(net, test_data)
print(f'Initial testing accuracy: {acc * 100}%')
