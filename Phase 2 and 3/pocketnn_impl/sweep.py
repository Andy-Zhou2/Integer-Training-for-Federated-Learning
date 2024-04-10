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
        'values': [200, 500, 1000, 2000, 5000, 10000]
    },
    'batch_size': {
        'distribution': 'int_uniform',
        'min': 10,
        'max': 30
    },
    'epochs': {
        'values': [1, 2, 3]
    },
    'shuffle_dataset_every_epoch': {
        'values': [True, False]
    }
}

# also set fixed parameters
parameters_dict.update({
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
})

sweep_config['parameters'] = parameters_dict

import pprint
pprint.pprint(sweep_config)

sweep_id = wandb.sweep(sweep_config, project='part ii diss', entity='wz337')