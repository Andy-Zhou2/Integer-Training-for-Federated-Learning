import wandb

sweep_config = {
    'method': 'random',
    'metric': {
        'name': 'max_acc',
        'goal': 'maximize'
    },
}

parameters_dict = {
    'lr_inv': {
        'distribution': 'int_uniform',
        'min': 400,
        'max': 2500
    },
    'batch_size': {
        'distribution': 'int_uniform',
        'min': 10,
        'max': 30
    },
    'epochs': {
        'distribution': 'int_uniform',
        'min': 1,
        'max': 20
    },
}

# also set fixed parameters
parameters_dict.update({
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
        'value': 100
    },
    'client_resources': {
        'value': {"num_cpus": 1, "num_gpus": 0.0}
    },
    'test_every_epoch': {
        'value': False
    },
    'train_ratio': {
        'value': 0.8
    },
    'fraction_fit': {
        'value': 1.0
    },
    'fraction_evaluate': {
        'value': 1.0
    },
    'global_seed': {
        'value': 123
    },
})

sweep_config['parameters'] = parameters_dict

import pprint
pprint.pprint(sweep_config)

sweep_id = wandb.sweep(sweep_config, project='part ii diss', entity='wz337')