import wandb
import pprint

sweep_config = {
    'method': 'random',
    'metric': {
        'name': 'max_pkt_acc',
        'goal': 'maximize'
    },
}

parameters_dict = {
    'dataset_dirichlet_alpha': {
        'values': [0.1, 1, 1000]
    },
    'train_ratio': {
        'values': [0.1, 0.5, 0.8, 1]
    },
    'pkt_epochs': {
        'distribution': 'int_uniform',
        'min': 1,
        'max': 10
    },
    'fp_weight_independence': {
        'values': [False, True]
    },
    'pkt_weight_independence': {
        'values': [False, True]
    },
    'pkt_params_weight': {   # need to set fp_params_weight manually in sweep agent
        'distribution': 'int_uniform',
        'min': 0,
        'max': 10
    },
    'fp_threshold_cid': {
        'distribution': 'int_uniform',
        'min': 9,
        'max': 89
    },
}

parameters_dict.update({
    'num_clients': {
        'value': 100
    },
    'dataset_name': {
        'value': 'mnist'
    },
    'num_rounds': {
        'value': 30
    },
    'client_resources': {
        'value': {
            'num_cpus': 1,
            'num_gpus': 0.0
        }
    },
    'lr': {
        'value': 0.1
    },
    'batch_size': {
        'value': 20
    },

    'fp_epochs': {
        'value': 1
    },
    'test_every_epoch': {
        'value': False
    },
    'num_fit_clients': {
        'value': 100
    },
    'fraction_evaluate': {
        'value': 0
    },
    'global_seed': {
        'value': 123
    },
    'model_name': {
        'value': 'mnist_default'
    },
    'use_wandb': {
        'value': True
    },
    'train_verbose': {
        'value': False
    },
    'gamma': {
        'value': 0.5
    },
    'step_size': {
        'value': 10
    },
    'lr_inv': {
        'value': 1000
    },
    'shuffle_dataset_every_epoch': {
        'value': True
    },
    'label_target_value': {
        'value': 15
    },

})

sweep_config['parameters'] = parameters_dict

pprint.pprint(sweep_config)

sweep_id = wandb.sweep(sweep_config, project='part ii diss', entity='wz337')
