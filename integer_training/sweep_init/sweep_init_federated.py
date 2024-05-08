import wandb
import pprint

sweep_config = {
    'method': 'grid',
    'metric': {
        'name': 'max_acc',
        'goal': 'maximize'
    },
}

model_names = []
for bw1 in [4, 6, 8, 10, 12, 16, 20]:
    for bw2 in [4, 6, 8, 10, 12, 16, 20]:
        for bw3 in [4, 6, 8, 10, 12, 16, 20]:
            cap1 = 2 ** (bw1 - 1) - 1
            cap2 = 2 ** (bw2 - 1) - 1
            cap3 = 2 ** (bw3 - 1) - 1
            model_names.append(f'mnist_default [{cap1},{cap2},{cap3}]')

parameters_dict = {
    'model_name': {
        'values': model_names
    },
}

# also set fixed parameters
parameters_dict.update({
    'dataset_dirichlet_alpha': {
        'value': 1
    },
    'fraction_fit': {
        'value': 0.15
    },
    'epochs': {
        'value': 1
    },
    'batch_size': {
        'value': 20
    },
    'lr_inv': {
        'value': 1000
    },
    'shuffle_dataset_every_epoch': {
        'value': True
    },
    'num_clients': {
        'value': 100
    },
    'dataset_name': {
        'value': 'mnist'
    },
    'num_rounds': {
        'value': 500
    },
    'client_resources': {
        'value': {"num_cpus": 1, "num_gpus": 0.0}
    },
    'test_every_epoch': {
        'value': False
    },
    'train_ratio': {
        'value': 1
    },
    'fraction_evaluate': {
        'value': 0
    },
    'global_seed': {
        'value': 123
    },
    'use_wandb': {
        'value': True
    },
    'train_verbose': {
        'value': False
    },
    'label_target_value': {
        'value': 15
    },
    'gamma_inv': {
        'value': 2
    },
    'gamma_step': {
        'value': 10
    },
})

sweep_config['parameters'] = parameters_dict

pprint.pprint(sweep_config)

sweep_id = wandb.sweep(sweep_config, project='part ii diss', entity='wz337')
