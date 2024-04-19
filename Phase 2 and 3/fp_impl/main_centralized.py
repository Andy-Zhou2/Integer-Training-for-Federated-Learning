import torch
from dataset import get_centralized_dataloader
from network import get_net
from train_evaluate import train_one_epoch, evaluate_model, train
import hydra
from omegaconf import DictConfig, OmegaConf
import logging
from utils_random import set_seed


@hydra.main(config_path='Configs/centralized', config_name='mnist', version_base='1.2')
def main(config: DictConfig):
    logging.info(OmegaConf.to_yaml(config))

    dataset = config.dataset
    batch_size = config.batch_size
    seed = config.seed

    train_config = {
        'epochs': config.epochs,
        'lr': config.lr,
        'gamma': config.gamma,
        'step_size': config.step_size,
        'verbose': config.verbose,
        'test_every_epoch': config.test_every_epoch,
        'weight_folder': config.weight_folder,
    }

    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = get_net(dataset + '_default').to(device)

    # Data loading
    train_loader, test_loader = get_centralized_dataloader(dataset, batch_size)

    data = {
        'train': train_loader,
        'test': test_loader
    }

    result = train(model, device, data, train_config)

    logging.info("Training complete:")
    logging.info(result)


if __name__ == '__main__':
    main()
