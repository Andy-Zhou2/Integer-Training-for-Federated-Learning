import wandb
import pprint

sweep_config = {
    'method': 'grid',
    'metric': {
        'name': 'mean_max_acc',
        'goal': 'maximize'
    },
}

parameters_dict = {
    'gamma_step': {
        'values': [1, 2, 3, 4, 5, 8, 10, 15, 20, 25, 30, 50]
    },
    'gamma_inv': {
        'values': [1.001, 1.1, 1.2, 1.5, 2, 2.5, 3.3, 5, 8, 10, 20, 50, 100]
    },
}

# also set fixed parameters
parameters_dict.update({
    'model_name': {
        'value': 'mnist_default'
    },
    'label_target_value': {
        'value': 15
    },
    'initial_lr_inv': {
        'value': 1000
    },
    'batch_size': {
        'value': 20
    },
    'shuffle_dataset_every_epoch': {
        'value': True
    },
    'dataset': {
        'value': 'mnist'
    },
    'test_every_epoch': {
        'value': True
    },
    'verbose': {
        'value': True
    },
    'epochs': {
        'value': 100
    },
    'print_hash_every_epoch': {
        'value': False
    },
    'initial_test': {
        'value': False
    },
})

sweep_config['parameters'] = parameters_dict

pprint.pprint(sweep_config)

sweep_id = wandb.sweep(sweep_config, project='part ii diss', entity='wz337')
