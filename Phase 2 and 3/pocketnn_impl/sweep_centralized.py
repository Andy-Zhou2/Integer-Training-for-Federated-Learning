import wandb
import os
from network import get_net
from train_evaluate import pktnn_train, pktnn_evaluate
from dataset import get_dataset
import logging
from utils_random import set_seed
import shutil


def agent_sweep():
    with wandb.init():
        # set logging level to info
        logging.basicConfig(level=logging.INFO)
        config = wandb.config

        dataset_name = config.dataset
        assert dataset_name in ['mnist', 'fashion_mnist']

        train_data, test_data = get_dataset(dataset_name)

        layer1_clip = 2 ** (config.layer_1_bw - 1) - 1
        layer2_clip = 2 ** (config.layer_2_bw - 1) - 1
        layer3_clip = 2 ** (config.layer_3_bw - 1) - 1
        model_name = f'mnist_default [{layer1_clip},{layer2_clip},{layer3_clip}]'

        custom_name = f'cent_lr{config.initial_lr_inv}_bs{config.batch_size}_model{model_name}'
        wandb.run.name = custom_name
        # create folder
        weight_folder = ''  # don't save for sweep

        logging.info(f'Creating model: {model_name}')
        net = get_net(model_name)

        seed = config.seed
        set_seed(seed)

        config = {
            'epochs': config.epochs,
            'batch_size': config.batch_size,
            'initial_lr_inv': config.initial_lr_inv,
            'weight_folder': weight_folder,
            'test_every_epoch': config.test_every_epoch,
            'print_hash_every_epoch': config.print_hash_every_epoch,
            'shuffle_dataset_every_epoch': config.shuffle_dataset_every_epoch,
            'verbose': config.verbose,
            'label_target_value': config.label_target_value,
        }

        pkt_data = {
            'train': train_data,
            'test': test_data
        }
        result = pktnn_train(net, pkt_data, config)
        logging.info(f'Train completed. Result: ')
        logging.info(result)

        for epoch in range(config['epochs']):
            report = {
                'loss': result['loss'][epoch],
                'train_accuracy': result['train_accuracy'][epoch],
            }
            if result['test_accuracy']:
                report['test_accuracy'] = result['test_accuracy'][epoch]
            wandb.log(report)

        max_acc = max(result['test_accuracy'])

        wandb.log({'max_acc': max_acc})


if __name__ == '__main__':
    wandb.agent(sweep_id='gf3jxv34', function=agent_sweep,
                project='part ii diss', entity='wz337', count=100)
