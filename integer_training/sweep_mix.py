import wandb
from src.fl.mix_fl import simulate


def agent_sweep():
    with wandb.init():
        config = wandb.config
        # pkt_params_weight and fp_params_weight are complementary
        config.fp_params_weight = 10 - config.pkt_params_weight
        hist = simulate(config)

        pkt_centralized_acc = hist.metrics_centralized['pkt_accuracy']
        max_acc = max([acc for _, acc in pkt_centralized_acc])

        wandb.log({'max_pkt_acc': max_acc})


if __name__ == '__main__':
    wandb.agent(sweep_id='ckaki4tj', function=agent_sweep,
                project='part ii diss', entity='wz337', count=100)
