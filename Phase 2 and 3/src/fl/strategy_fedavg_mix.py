from typing import Dict, List, Optional, Tuple
from typing import Callable, Union
import flwr as fl
import numpy as np
import logging
from flwr.server.strategy.aggregate import aggregate as aggregate_fp
from .strategy_fedavg_int import aggregate_int

from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    Parameters,
    Scalar,
    parameters_to_ndarrays,
    ndarrays_to_parameters
)
from flwr.common.typing import NDArrays
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy


class FedAvgMix(fl.server.strategy.Strategy):
    def __init__(
            self,
            fp_config: Dict[str, Scalar],
            pkt_config: Dict[str, Scalar],
            fp_client_id_threshold: int,
            eval_fn: Callable[[NDArrays], Optional[Tuple[float, Dict[str, Scalar]]]],
            initial_parameters: Parameters,
            min_fit_clients: int = 2,
            fp_weight_independence: bool = False,
            pkt_weight_independence: bool = False,
            fp_weight: int = 1,
            pkt_weight: int = 1,
    ) -> None:
        """
        Does not support client-side evaluation (fraction_evaluate=0).

        @:param fp_client_id_threshold: Client id that <= this value will be considered as FP clients.
        @:param fp_weight_independence: If True, the weights of FP clients will not be affected by PKT clients.
        @:param pkt_weight_independence: If True, the weights of PKT clients will not be affected by FP clients.
        @:param fp_weight: The weight of FP clients in the final aggregation.
            This will be considered in conjunction with pkt_weight.
        @:param pkt_weight: The weight of PKT clients in the final aggregation.
            This will be considered in conjunction with fp_weight.
        """
        super().__init__()
        self.min_fit_clients = min_fit_clients
        self.initial_parameters = initial_parameters
        self.fp_client_id_threshold = fp_client_id_threshold
        self.fp_config = fp_config
        self.pkt_config = pkt_config
        self.eval_fn = eval_fn
        self.fp_weight_independence = fp_weight_independence
        self.pkt_weight_independence = pkt_weight_independence
        self.fp_params_weight = fp_weight
        self.pkt_params_weight = pkt_weight

    def __repr__(self) -> str:
        return "FedAvgMix"

    def initialize_parameters(
            self, client_manager: ClientManager
    ) -> Optional[Parameters]:
        """Initialize global model parameters."""
        initial_parameters = self.initial_parameters
        self.initial_parameters = None  # Don't keep initial parameters in memory
        return initial_parameters

    def configure_fit(
            self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""

        clients = client_manager.sample(
            num_clients=self.min_fit_clients, min_num_clients=self.min_fit_clients
        )
        params = parameters_to_ndarrays(parameters)
        fp_params = ndarrays_to_parameters(params[:len(params) // 2])
        pkt_params = ndarrays_to_parameters(params[len(params) // 2:])

        fit_configurations = []
        for client in clients:
            if int(client.cid) <= self.fp_client_id_threshold:
                fit_configurations.append((client, FitIns(fp_params, self.fp_config)))
            else:
                fit_configurations.append((client, FitIns(pkt_params, self.pkt_config)))

        return fit_configurations

    def aggregate_fit(
            self,
            server_round: int,
            results: List[Tuple[ClientProxy, FitRes]],
            failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:

        # collect weights and num_samples
        fp_weights_samples = []  # stores pairs of (weight, number of samples)
        pkt_weights_samples = []

        for client, result in results:
            cid = int(client.cid)
            if cid <= self.fp_client_id_threshold:
                fp_weights_samples.append((parameters_to_ndarrays(result.parameters), result.num_examples))
            else:
                pkt_weights_samples.append((parameters_to_ndarrays(result.parameters), result.num_examples))

        # calculate mean and stddev
        fp_means = []
        fp_stds = []
        for i in range(len(fp_weights_samples[0][0])):
            fp_means.append(np.mean([w[i] for w, _ in fp_weights_samples]))
            fp_stds.append(np.std([w[i] for w, _ in fp_weights_samples]))
        pkt_means = []
        pkt_stds = []
        for i in range(len(pkt_weights_samples[0][0])):
            pkt_means.append(np.mean([w[i] for w, _ in pkt_weights_samples]))
            pkt_stds.append(np.std([w[i] for w, _ in pkt_weights_samples]))
        logging.info(f'fp means: {fp_means}')
        logging.info(f'fp stds: {fp_stds}')
        logging.info(f'pkt means: {pkt_means}')
        logging.info(f'pkt stds: {pkt_stds}')

        # calculate fp and pkt individually
        fp_avg_weight = aggregate_fp(fp_weights_samples)
        pkt_avg_weight = aggregate_int(
            [(client, result) for client, result in results if int(client.cid) > self.fp_client_id_threshold])

        # map pkt to fp and align the shape
        pkt_to_fp_weight = [
            (pkt_avg_weight[i] - pkt_means[i]) / pkt_stds[i] * fp_stds[i] + fp_means[i]
            for i in range(len(fp_avg_weight))
        ]
        for i in range(len(pkt_to_fp_weight)):
            w = pkt_to_fp_weight[i]

            if np.isnan(w).any():  # if w invalid, use fp results
                # nan usually happens when the std is 0 and divided by 0 occurs
                w = fp_avg_weight[i]
            if i % 2 == 1:  # bias, 1xdim
                pkt_to_fp_weight[i] = w[0]
            else:  # weight matrix, dim_out x dim_in
                pkt_to_fp_weight[i] = w.T

        # aggregate (take average) and then align back to pkt format
        mix_weight = aggregate_fp([(pkt_to_fp_weight, self.pkt_params_weight), (fp_avg_weight, self.fp_params_weight)])

        mix_to_pkt_weight = [
            (mix_weight[i] - fp_means[i]) / fp_stds[i] * pkt_stds[i] + pkt_means[i]
            for i in range(len(mix_weight))
        ]
        for i in range(len(mix_to_pkt_weight)):
            w = mix_to_pkt_weight[i]
            if i % 2 == 1:  # bias
                mix_to_pkt_weight[i] = np.expand_dims(w, axis=0)
            else:  # weight
                mix_to_pkt_weight[i] = w.T

        mix_to_pkt_weight = [w.astype(np.int64) for w in mix_to_pkt_weight]

        result_weight = []
        result_weight.extend(fp_avg_weight if self.fp_weight_independence else mix_weight)
        result_weight.extend(pkt_avg_weight if self.pkt_weight_independence else mix_to_pkt_weight)

        return ndarrays_to_parameters(result_weight), {}

    def configure_evaluate(
            self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """Does not support client-side evaluation"""
        return []

    def aggregate_evaluate(
            self,
            server_round: int,
            results: List[Tuple[ClientProxy, EvaluateRes]],
            failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        raise NotImplementedError("FedDC does not support client-side evaluation")

    def evaluate(
            self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        parameters_ndarrays = parameters_to_ndarrays(parameters)
        return self.eval_fn(parameters_ndarrays)
