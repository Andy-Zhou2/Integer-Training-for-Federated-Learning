import os
from src.pktnn.pkt_network import get_net
from src.pktnn.train_evaluate import pktnn_train, pktnn_evaluate
from src.dataset.pkt_dataset import get_centralized_dataloader_pkt
from distutils.dir_util import copy_tree, remove_tree
import numpy as np
import pprint

print('Loading data')

dataset_name = 'mnist'  # mnist or fashion_mnist
assert dataset_name in ['mnist', 'fashion_mnist']

train_data, test_data = get_centralized_dataloader_pkt(dataset_name)
result = dict()

bws_backup = [32767] * 3


for bw in range(2, 16):
    for position in range(3):
        clip = 2 ** (bw-1) - 1

        bws = bws_backup.copy()
        bws[position] = clip
        weight_folder = rf'C:\Users\zhouw\Desktop\Integer-Training-for-Federated-Learning\data\weights\mnist_mnist_default [{bws[0]},{bws[1]},{bws[2]}]'
        if not os.path.exists(weight_folder):
            print(f'Does not exist {bws}')
            continue
        print(f'Working on {bws}')

        # os.mkdir('./activations')
        # print('Creating model')
        accs = [0.098]
        for epoch in range(1, 41):
            print(epoch)
            net = get_net(dataset_name + '_default')
            net.load(os.path.join(weight_folder, f'epoch_{epoch}.npz'))

            net_dict = np.load(os.path.join(weight_folder, f'epoch_{epoch}.npz'), allow_pickle=True)
            # for key in net_dict:
            #     weight = net_dict[key]
            #     print(f'{max(abs(weight.max()), abs(weight.min()))}', end='\t')
            #     # print(f'{key}: {np.percentile(net_dict[key], np.arange(0, 101, 50))}')
            # print()

            # initial testing
            # print('Doing Inference...')
            # acc = pktnn_evaluate(net, train_data)
            # print(f'Initial training accuracy: {acc * 100}%')
            acc = pktnn_evaluate(net, test_data)
            accs.append(acc)
            print(f'Initial testing accuracy: {acc * 100}%')

            # transfer contents in activations to save_activations/bw
            # os.mkdir(f'./save_activations/{bw}')
            # copy_tree('./activations', f'./save_activations/{bw}')
            # remove_tree('./activations')

        result[tuple(bws)] = accs
        print(result)
pprint.pprint(result)

with open(f'./sensitivity.txt', 'w') as f:
    pprint.pprint(result, f)