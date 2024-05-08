from src.fl.fp_fl import simulate
from omegaconf import DictConfig, OmegaConf
import hydra
from flwr.common.logger import log
import logging


@hydra.main(config_path='configs/fp/fl', config_name='mnist', version_base='1.2')
def main(config: DictConfig):
    log(logging.INFO, OmegaConf.to_yaml(config))

    # Convert dictionary to SimpleNamespace to allow dot access
    hist = simulate(config)
    log(logging.INFO, hist)

    centralized_acc = hist.metrics_centralized['accuracy']
    final_round_acc = centralized_acc[-1][1]
    max_acc = max([acc for _, acc in centralized_acc])

    log(logging.INFO, f'Final round accuracy: {final_round_acc}')
    log(logging.INFO, f'Max accuracy: {max_acc}')


if __name__ == '__main__':
    main()