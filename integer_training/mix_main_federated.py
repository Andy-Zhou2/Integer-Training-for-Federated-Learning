from src.fl.mix_fl import simulate
from omegaconf import DictConfig, OmegaConf
import hydra
from flwr.common.logger import log
import logging


@hydra.main(config_path='configs/mix', config_name='mnist', version_base='1.2')
def main(config: DictConfig):
    log(logging.INFO, OmegaConf.to_yaml(config))

    # Convert dictionary to SimpleNamespace to allow dot access
    hist = simulate(config)
    log(logging.INFO, hist)


if __name__ == '__main__':
    main()
