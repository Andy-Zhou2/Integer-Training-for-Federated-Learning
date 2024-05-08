import numpy as np
from typing import List, Dict, Any, Tuple
import random
from collections import defaultdict
import numbers
import flwr as fl
from flwr.common.typing import NDArrays
import wandb

from .strategy_fedavg_int import FedAvgInt
from ..pktnn.pkt_network import get_net, PktNet
from ..dataset.pkt_dataset import DatasetTuple, load_federated_dataset_pkt, ClientDatasetPkt
from ..pktnn.train_evaluate import pktnn_train, pktnn_evaluate
from ..utils.utils_random import generate_rng, DeterministicClientManager, set_seed


class FlowerClient(fl.client.NumPyClient):
    def __init__(self, net: PktNet, client_dataset: ClientDatasetPkt, cid: str, seed: int):
        self.net = net
        self.client_dataset = client_dataset
        self.cid = cid
        self.seed = seed

    def get_parameters(self, config):
        return self.net.get_parameters()

    def fit(self, parameters: NDArrays, config: Dict[str, Any]):
        random.seed(self.seed)
        np.random.seed(self.seed)

        self.net.set_parameters(parameters)
        pktnn_train(self.net, self.client_dataset, config=config | {'cid': self.cid})
        params = self.net.get_parameters()

        len_train_data = self.client_dataset['train'][0].shape[0]
        return params, len_train_data, {'cid': self.cid}

    def evaluate(self, parameters: NDArrays, config: Dict[str, Any]):
        if self.client_dataset['test'] is None:
            return float('NAN'), 1, {"accuracy": float('NAN'),
                                     'cid': self.cid}  # return 1 as the number of data to avoid division by 0
        self.net.set_parameters(parameters)
        accuracy = pktnn_evaluate(self.net, self.client_dataset['test'])
        len_val_data = self.client_dataset['test'][0].shape[0]

        return float('NAN'), len_val_data, {"accuracy": accuracy, 'cid': self.cid}


def client_fn(cid: str, model_name: str, client_datasets: List[ClientDatasetPkt], seed: int):
    # Load model
    net = get_net(model_name)
    cid_int = int(cid)
    client_dataset = client_datasets[cid_int]

    return FlowerClient(net, client_dataset, cid, seed).to_client()


def aggregate_weighted_average(metrics: list[tuple[int, dict]]) -> dict:
    """Combine results from multiple clients.

    Args:
        metrics (list[tuple[int, dict]]): collected clients metrics

    Returns
    -------
        dict: result dictionary containing the aggregate of the metrics passed.
    """
    average_dict: dict = defaultdict(list)
    total_examples: int = 0
    for num_examples, metrics_dict in metrics:
        for key, val in metrics_dict.items():
            if isinstance(val, numbers.Number):
                average_dict[key].append((num_examples, val))
        total_examples += num_examples
    return {
        key: {
            "avg": float(
                sum([num_examples * m for num_examples, m in val])
                / float(total_examples)
            ),
            "all": val,
        }
        for key, val in average_dict.items()
    }


def _on_fit_config_fn(client_train_config: dict, server_round: int):
    return client_train_config | {"server_round": server_round}


def _on_evaluate_config_fn(server_round: int):
    return {"server_round": server_round}


def federated_evaluation_function(model_name: str, test_dataset: DatasetTuple,
                                  server_round: int, parameters: NDArrays, fed_eval_config: Dict[str, Any],
                                  use_wandb: bool = False) -> Tuple[float, Dict[str, Any]]:
    """returns (loss, dict of results)"""
    net = get_net(model_name)
    net.set_parameters(parameters)
    accuracy = pktnn_evaluate(net, test_dataset)
    if use_wandb:
        wandb.log({"accuracy": accuracy})
    return float('NAN'), {"accuracy": accuracy}


def simulate(config):
    global_seed = config.global_seed
    set_seed(global_seed)  # set seed before generating rng and client_seed

    client_seed = np.random.randint(0, 2 ** 31 - 1)
    num_clients = config.num_clients
    dataset_name = config.dataset_name
    dataset_dirichlet_alpha = config.dataset_dirichlet_alpha
    num_rounds = config.num_rounds
    client_resources = config.client_resources
    client_train_config = {
        'epochs': config.epochs,
        'batch_size': config.batch_size,
        'initial_lr_inv': config.lr_inv,
        'weight_folder': '',  # empty string: don't save
        'test_every_epoch': config.test_every_epoch,
        'print_hash_every_epoch': False,
        'shuffle_dataset_every_epoch': config.shuffle_dataset_every_epoch,
        'verbose': config.train_verbose,
        'label_target_value': config.label_target_value,
        'gamma_inv': config.gamma_inv,
        'gamma_step': config.gamma_step,
    }
    train_ratio = config.train_ratio  # proportion of the training set used for training (the rest for validation)
    fraction_fit = config.fraction_fit
    fraction_evaluate = config.fraction_evaluate
    model_name = config.model_name  # model name to be used, such as mnist_default or custom [100, 100]
    use_wandb = config.use_wandb  # report each round accuracy to wandb if True

    _, client_cid_rng, _ = generate_rng(global_seed)

    client_datasets, test_dataset = load_federated_dataset_pkt(dataset_name, dataset_dirichlet_alpha, num_clients, train_ratio)

    client_manager = DeterministicClientManager(client_cid_rng, enable_resampling=False)

    strategy = FedAvgInt(
        fraction_fit=fraction_fit,
        fraction_evaluate=fraction_evaluate,
        evaluate_metrics_aggregation_fn=aggregate_weighted_average,
        on_fit_config_fn=lambda server_round: _on_fit_config_fn(client_train_config, server_round),
        on_evaluate_config_fn=_on_evaluate_config_fn,
        evaluate_fn=lambda server_round, parameters, fed_eval_config: federated_evaluation_function(
            model_name, test_dataset, server_round, parameters, fed_eval_config, use_wandb),
        inplace=False
    )

    # Start simulation
    hist = fl.simulation.start_simulation(
        client_fn=lambda cid: client_fn(cid, model_name, client_datasets, client_seed),
        num_clients=num_clients,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
        client_resources=client_resources,
        client_manager=client_manager
    )

    return hist
