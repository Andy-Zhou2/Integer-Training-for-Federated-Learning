import os
from network import get_net
from train_evaluate import pktnn_train, pktnn_evaluate
from dataset import get_dataset
import hydra
from omegaconf import DictConfig, OmegaConf
import logging
from utils_random import set_seed


@hydra.main(config_path='Configs/centralized', config_name='mnist', version_base='1.2')
def main(config: DictConfig):
    logging.info(OmegaConf.to_yaml(config))
    logging.info('Loading data')

    dataset_name = config.dataset
    assert dataset_name in ['mnist', 'fashion_mnist']

    train_data, test_data = get_dataset(dataset_name)

    # create folder
    weight_folder = config.weight_folder
    os.makedirs(weight_folder, exist_ok=True)

    logging.info('Creating model')
    net = get_net(dataset_name + '_default')

    # initial testing
    # logging.info('Initial testing')
    # acc = pktnn_evaluate(net, train_data)
    # logging.info(f'Initial training accuracy: {acc * 100}%')
    # acc = pktnn_evaluate(net, test_data)
    # logging.info(f'Initial testing accuracy: {acc * 100}%')

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


if __name__ == '__main__':
    main()
