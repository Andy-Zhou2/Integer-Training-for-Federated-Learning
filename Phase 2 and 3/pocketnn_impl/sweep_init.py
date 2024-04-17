import wandb
import pprint

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
        'min': 100,
        'max': 3000
    },
    'batch_size': {
        'distribution': 'int_uniform',
        'min': 10,
        'max': 40
    },
    'epochs': {
        'distribution': 'int_uniform',
        'min': 1,
        'max': 20
    },
    'dataset_dirichlet_alpha': {
        'values': [0.1, 1, 1000]
    },
    'model_name': {
        'values': ['custom []',
                   'custom [50]', 'custom [100]', 'custom [200]', 'custom [100, 100]',
                   'custom [200, 100]', 'custom [200, 100, 50]', 'custom [400, 200, 100]',
                   'custom [400, 200, 100, 50]',
                   'custom [400, 200, 200, 100, 50]',
                   'fashion_mnist_default', 'mnist_default']
    },
    'fraction_fit': {
        'values': [0.15, 1]
    }
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
        'value': 'fashion_mnist'
    },
    'num_rounds': {
        'value': 40
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
    }
})

sweep_config['parameters'] = parameters_dict

pprint.pprint(sweep_config)

sweep_id = wandb.sweep(sweep_config, project='part ii diss', entity='wz337')
