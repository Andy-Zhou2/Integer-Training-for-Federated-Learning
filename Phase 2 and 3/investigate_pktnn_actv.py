import os
from src.pktnn.pkt_network import get_net
from src.pktnn.train_evaluate import pktnn_train, pktnn_evaluate
from src.dataset.pkt_dataset import get_centralized_dataloader_pkt
from distutils.dir_util import copy_tree, remove_tree
import numpy as np

print('Loading data')

dataset_name = 'mnist'  # mnist or fashion_mnist
assert dataset_name in ['mnist', 'fashion_mnist']

train_data, test_data = get_centralized_dataloader_pkt(dataset_name)

for bw in range(2, 64):

    # create folder
    clip = 2 ** (bw-1) - 1
    weight_folder = rf'C:\Users\zhouw\Desktop\data\weights\mnist_mnist_default [{clip},{clip},{clip}]'
    if not os.path.exists(weight_folder):
        print(f'Does not exist {bw}')
        continue
    print(f'Working on {bw}')

    # os.mkdir('./activations')
    print('Creating model')
    net = get_net(dataset_name + '_default')
    net.load(os.path.join(weight_folder, 'epoch_40.npz'))

    net_dict = np.load(os.path.join(weight_folder, 'epoch_40.npz'), allow_pickle=True)
    for key in net_dict:
        print(f'{key}: {np.percentile(net_dict[key], np.arange(0, 101, 50))}')

    # initial testing
    print('Doing Inference...')
    # acc = pktnn_evaluate(net, train_data)
    # print(f'Initial training accuracy: {acc * 100}%')
    acc = pktnn_evaluate(net, test_data)
    print(f'Initial testing accuracy: {acc * 100}%')

    # transfer contents in activations to save_activations/bw
    # os.mkdir(f'./save_activations/{bw}')
    # copy_tree('./activations', f'./save_activations/{bw}')
    # remove_tree('./activations')
