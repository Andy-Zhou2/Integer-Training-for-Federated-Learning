import wandb
import logging
from src.pktnn.pkt_network import get_net
from src.pktnn.train_evaluate import pktnn_train
from src.dataset.pkt_dataset import get_centralized_dataloader_pkt
from src.utils.utils_random import set_seed
import numpy as np


def agent_sweep():
    with wandb.init():
        # set logging level to info
        logging.basicConfig(level=logging.INFO)
        config = wandb.config

        dataset_name = config.dataset
        assert dataset_name in ['mnist', 'fashion_mnist']

        train_data, test_data = get_centralized_dataloader_pkt(dataset_name)

        model_name = config.model_name

        custom_name = f'step_{config.gamma_step}_inv_{config.initial_lr_inv}'
        wandb.run.name = custom_name

        weight_folder = ''  # don't save for sweep

        logging.info(f'Creating model: {model_name}')
        net = get_net(model_name)

        max_accs = []

        for seed in [1, 2, 3, 4, 5]:
            set_seed(seed)

            train_config = {
                'epochs': config.epochs,
                'batch_size': config.batch_size,
                'initial_lr_inv': config.initial_lr_inv,
                'weight_folder': weight_folder,
                'test_every_epoch': config.test_every_epoch,
                'print_hash_every_epoch': config.print_hash_every_epoch,
                'shuffle_dataset_every_epoch': config.shuffle_dataset_every_epoch,
                'verbose': config.verbose,
                'label_target_value': config.label_target_value,
                'gamma_inv': config.gamma_inv,
                'gamma_step': config.gamma_step,
            }

            pkt_data = {
                'train': train_data,
                'test': test_data
            }
            result = pktnn_train(net, pkt_data, train_config)
            logging.info(f'Train completed. Result: ')
            logging.info(result)

            for epoch in range(train_config['epochs']):
                report = {
                    f'loss_{seed}': result['loss'][epoch],
                    f'train_accuracy_{seed}': result['train_accuracy'][epoch],
                }
                if result['test_accuracy']:
                    report[f'test_accuracy_{seed}'] = result['test_accuracy'][epoch]
                wandb.log(report)

            max_acc = max(result['test_accuracy'])
            max_accs.append(max_acc)

            wandb.log({f'max_acc_{seed}': max_acc})

        mean_max_acc = np.mean(max_accs)
        std_max_acc = np.std(max_accs)
        wandb.log({'mean_max_acc': mean_max_acc, 'std_max_acc': std_max_acc})


if __name__ == '__main__':
    wandb.agent(sweep_id='i5imkhh2', function=agent_sweep,
                project='part ii diss', entity='wz337', count=100)
