import wandb
from src.fl.pkt_fl import simulate


def agent_sweep():
    with wandb.init():
        custom_name = f'p{wandb.config.fraction_fit}_lr{wandb.config.lr_inv}_bs{wandb.config.batch_size}_ep{wandb.config.epochs}_alpha{wandb.config.dataset_dirichlet_alpha}_model{wandb.config.model_name}'
        wandb.run.name = custom_name

        config = wandb.config
        hist = simulate(config)

        centralized_acc = hist.metrics_centralized['accuracy']
        final_round_acc = centralized_acc[-1][1]
        max_acc = max([acc for _, acc in centralized_acc])

        wandb.log({'final_round_acc': final_round_acc, 'max_acc': max_acc})


if __name__ == '__main__':
    wandb.agent(sweep_id='s2lg6h0d', function=agent_sweep,
                project='part ii diss', entity='wz337', count=100)
