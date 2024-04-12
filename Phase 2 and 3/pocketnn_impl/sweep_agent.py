from fl import simulate
import wandb


def agent_sweep():
    with wandb.init():
        # custom name containing lr, bs, epochs, shuffle
        custom_name = f'lr{wandb.config.lr_inv}_bs{wandb.config.batch_size}_ep{wandb.config.epochs}_sh_{wandb.config.shuffle_dataset_every_epoch}'
        wandb.run.name = custom_name

        config = wandb.config
        hist = simulate(config)

        centralized_acc = hist.metrics_centralized['accuracy']
        final_round_acc = centralized_acc[-1][1]
        max_acc = max([acc for _, acc in centralized_acc])

        wandb.log({'final_round_acc': final_round_acc, 'max_acc': max_acc})


if __name__ == '__main__':
    wandb.agent(sweep_id='9fv4290c', function=agent_sweep,
                project='part ii diss', entity='wz337', count=150)
