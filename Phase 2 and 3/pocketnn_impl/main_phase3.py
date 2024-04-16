from fl import simulate
from types import SimpleNamespace
from omegaconf import DictConfig, OmegaConf
import hydra


@hydra.main(config_path='Configs/FL', config_name='mnist', version_base='1.2')
def main(config: DictConfig):
    print(OmegaConf.to_yaml(config))

    # Convert dictionary to SimpleNamespace to allow dot access
    hist = simulate(config)
    print(hist)

    centralized_acc = hist.metrics_centralized['accuracy']
    final_round_acc = centralized_acc[-1][1]
    max_acc = max([acc for _, acc in centralized_acc])

    print(f'Final round accuracy: {final_round_acc}')
    print(f'Max accuracy: {max_acc}')


    # also report the first round number when acc >= threshold
    threshold = 0.95
    first_round = next(round for round, acc in centralized_acc if acc >= threshold)
    print(f'First round with accuracy >= {threshold}: {first_round}')


if __name__ == '__main__':
    main()