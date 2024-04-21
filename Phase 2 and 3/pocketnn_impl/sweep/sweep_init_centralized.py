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
    'initial_lr_inv': {
        'distribution': 'int_uniform',
        'min': 100,
        'max': 3000
    },
    'batch_size': {
        'distribution': 'int_uniform',
        'min': 2,
        'max': 40
    },
    'layer_1_bw': {  # using mnist_default
        'distribution': 'int_uniform',
        'min': 2,
        'max': 16
    },
    'layer_2_bw': {
        'distribution': 'int_uniform',
        'min': 2,
        'max': 16
    },
    'layer_3_bw': {
        'distribution': 'int_uniform',
        'min': 1,
        'max': 16
    },
    'label_target_value': {
        'values': [1, 3, 7, 15, 31, 63, 127, 255]
    }
}

# also set fixed parameters
parameters_dict.update({
    'shuffle_dataset_every_epoch': {
        'value': True
    },
    'dataset': {
        'value': 'fashion_mnist'
    },
    'test_every_epoch': {
        'value': True
    },
    'seed': {
        'value': 123
    },
    'verbose': {
        'value': True
    },
    'epochs': {
        'value': 50
    },
    'print_hash_every_epoch': {
        'value': False
    },
})

sweep_config['parameters'] = parameters_dict

pprint.pprint(sweep_config)

sweep_id = wandb.sweep(sweep_config, project='part ii diss', entity='wz337')
