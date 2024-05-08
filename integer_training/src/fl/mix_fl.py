import numpy as np
from typing import List, Dict, Any, Tuple
import wandb
import torch
from torch.utils.data import DataLoader
import flwr as fl
from flwr.common.typing import NDArrays
from flwr.common import ndarrays_to_parameters

from ..fp.network import get_net as fp_get_net
from ..pktnn.pkt_network import get_net as pkt_get_net
from ..dataset.dataset_core import DatasetTuple
from ..dataset.fp_dataset import load_federated_dataset_fp, ClientDatasetFP
from ..dataset.pkt_dataset import load_federated_dataset_pkt, ClientDatasetPkt
from ..utils.utils_random import generate_rng, DeterministicClientManager, set_seed
from .fp_fl import FlowerClient as FPFlowerClient, federated_evaluation_function as fp_federated_evaluation_function, \
    get_parameters as fp_get_parameters
from .pkt_fl import FlowerClient as PKTFlowerClient, federated_evaluation_function as pkt_federated_evaluation_function
from .strategy_fedavg_mix import FedAvgMix

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def client_fn(cid: str, model_name: str, fp_datasets: List[ClientDatasetFP], pkt_datasets: List[ClientDatasetPkt],
              seed: int, fp_threshold: int) -> fl.client.Client:
    set_seed(seed)

    cid_int = int(cid)
    if cid_int <= fp_threshold:
        net = fp_get_net(model_name).to(device)
        client_dataset = fp_datasets[cid_int]
        return FPFlowerClient(net, client_dataset, cid, seed).to_client()
    else:
        net = pkt_get_net(model_name)
        client_dataset = pkt_datasets[cid_int]
        return PKTFlowerClient(net, client_dataset, cid, seed).to_client()


def federated_evaluation_function(model_name: str, fp_test_dataset: DataLoader, pkt_test_dataset: DatasetTuple,
                                  server_round: int, parameters: NDArrays, fed_eval_config: Dict[str, Any],
                                  use_wandb: bool = False) -> Tuple[float, Dict[str, Any]]:
    fp_parameters = parameters[:len(parameters) // 2]  # the first half of the parameters are FP weights
    fp_loss, fp_acc = fp_federated_evaluation_function(model_name, fp_test_dataset, server_round, fp_parameters,
                                                       fed_eval_config, use_wandb=False)
    fp_acc = fp_acc['accuracy']

    pkt_parameters = parameters[len(parameters) // 2:]  # the second half of the parameters are PKT weights
    pkt_loss, pkt_acc = pkt_federated_evaluation_function(model_name, pkt_test_dataset, server_round, pkt_parameters,
                                                          fed_eval_config, use_wandb=False)
    pkt_acc = pkt_acc['accuracy']

    if use_wandb:
        wandb.log({'fp_accuracy': fp_acc, 'pkt_accuracy': pkt_acc})

    return (fp_loss + pkt_loss) / 2, {'fp_accuracy': fp_acc, 'pkt_accuracy': pkt_acc}  # pkt_loss is NAN


def simulate(config):
    global_seed = config.global_seed
    set_seed(global_seed)  # set seed before generating rng and client_seed

    client_seed = np.random.randint(0, 2 ** 31 - 1)
    num_clients = config.num_clients
    dataset_name = config.dataset_name
    dataset_dirichlet_alpha = config.dataset_dirichlet_alpha
    num_rounds = config.num_rounds
    client_resources = config.client_resources
    batch_size = config.batch_size
    fp_train_config = {
        'epochs': config.fp_epochs,
        'lr': config.lr,
        'gamma': config.gamma,
        'step_size': config.step_size,
        'test_every_epoch': config.test_every_epoch,
        'verbose': config.train_verbose,
        'weight_folder': '',  # don't save weights
    }
    pkt_train_config = {
        'epochs': config.pkt_epochs,
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
    model_name = config.model_name  # model name to be used, such as mnist_default or custom [100, 100]
    use_wandb = config.use_wandb  # report each round accuracy to wandb if True
    num_fit_clients = config.num_fit_clients
    fp_threshold_cid = config.fp_threshold_cid
    fp_weight_independence = config.fp_weight_independence
    pkt_weight_independence = config.pkt_weight_independence
    fp_params_weight = config.fp_params_weight
    pkt_params_weight = config.pkt_params_weight

    _, client_cid_rng, _ = generate_rng(global_seed)

    client_datasets_fp, test_dataset_fp = load_federated_dataset_fp(dataset_name, dataset_dirichlet_alpha, num_clients,
                                                                    train_ratio, batch_size, shuffle=True)
    client_datasets_pkt, test_dataset_pkt = load_federated_dataset_pkt(dataset_name, dataset_dirichlet_alpha,
                                                                       num_clients, train_ratio)

    client_manager = DeterministicClientManager(client_cid_rng, enable_resampling=False)

    # get initial parameters, ensuring correct layout
    init_net_fp = fp_get_net(model_name)
    init_parameters_fp = fp_get_parameters(init_net_fp)
    init_net_pkt = pkt_get_net(model_name)
    init_parameters_pkt = init_net_pkt.get_parameters()
    init_parameters = init_parameters_fp + init_parameters_pkt
    init_parameters = ndarrays_to_parameters(init_parameters)

    strategy = FedAvgMix(
        fp_config=fp_train_config,
        pkt_config=pkt_train_config,
        fp_client_id_threshold=fp_threshold_cid,
        eval_fn=lambda parameters: federated_evaluation_function(model_name, test_dataset_fp, test_dataset_pkt, -1,
                                                                 parameters, {}, use_wandb),
        min_fit_clients=num_fit_clients,
        initial_parameters=init_parameters,
        fp_weight_independence=fp_weight_independence,
        pkt_weight_independence=pkt_weight_independence,
        fp_weight=fp_params_weight,
        pkt_weight=pkt_params_weight
    )

    # Start simulation
    hist = fl.simulation.start_simulation(
        client_fn=lambda cid: client_fn(cid, model_name, client_datasets_fp, client_datasets_pkt, client_seed,
                                        fp_threshold_cid),
        num_clients=num_clients,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
        client_resources=client_resources,
        client_manager=client_manager
    )

    return hist
