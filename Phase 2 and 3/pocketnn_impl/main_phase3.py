from fl import simulate
from types import SimpleNamespace

config = {
    'num_clients': 100,
    'dataset_name': 'mnist',
    'num_rounds': 100,
    'client_resources': {"num_cpus": 1, "num_gpus": 0.0},
    'lr_inv': 1000,
    'batch_size': 15,
    'epochs': 10,
    'shuffle_dataset_every_epoch': True,
    'test_every_epoch': False,
    'train_ratio': 1,
    'fraction_fit': 1.0,
    'fraction_evaluate': 1.0,
    'global_seed': 123,
    'dataset_dirichlet_alpha': 1000,
}

# Convert dictionary to SimpleNamespace to allow dot access
config = SimpleNamespace(**config)
hist = simulate(config)
print(hist)
