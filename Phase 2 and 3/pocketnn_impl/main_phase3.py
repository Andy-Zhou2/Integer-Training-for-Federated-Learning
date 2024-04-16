from fl import simulate
from types import SimpleNamespace
from omegaconf import DictConfig, OmegaConf
import hydra
from flwr.common.logger import log


@hydra.main(config_path='Configs/FL', config_name='mnist', version_base='1.2')
def main(config: DictConfig):
    log(OmegaConf.to_yaml(config))

    # Convert dictionary to SimpleNamespace to allow dot access
    hist = simulate(config)
    log(hist)

    centralized_acc = hist.metrics_centralized['accuracy']
    final_round_acc = centralized_acc[-1][1]
    max_acc = max([acc for _, acc in centralized_acc])

    log(f'Final round accuracy: {final_round_acc}')
    log(f'Max accuracy: {max_acc}')


if __name__ == '__main__':
    main()