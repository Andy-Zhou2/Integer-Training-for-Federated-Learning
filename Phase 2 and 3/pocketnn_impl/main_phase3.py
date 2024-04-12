from fl import simulate
from types import SimpleNamespace

config = {
    'num_clients': 5,
    'dataset_name': 'fashion_mnist',
    'num_rounds': 20,
    'client_resources': {"num_cpus": 1, "num_gpus": 0.0},
    'lr_inv': 1000,
    'batch_size': 20,
    'epochs': 10,
    'shuffle_dataset_every_epoch': False,
    'test_every_epoch': False,
    'train_ratio': 0.8,
    'fraction_fit': 1.0,
    'fraction_evaluate': 1.0
}

# Convert dictionary to SimpleNamespace to allow dot access
config = SimpleNamespace(**config)
hist = simulate(config)
print(hist)
