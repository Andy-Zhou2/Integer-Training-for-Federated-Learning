import numpy as np
import flwr as fl
from typing import Callable, Dict, List, Optional, Tuple, Union

from flwr.common import (
    FitRes,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
    NDArrays
)
from flwr.server.client_proxy import ClientProxy
import logging
from logging import WARNING
from flwr.common.logger import log
from calc_util import truncate_divide


def aggregate(results: List[Tuple[ClientProxy, FitRes]]) -> NDArrays:
    # Sum total examples to use as a common multiplier to avoid floating-point division
    num_examples_total = sum(fit_res.num_examples for (_, fit_res) in results)

    initial_params = parameters_to_ndarrays(results[0][1].parameters)
    aggregated_params = [np.zeros_like(param, dtype=np.int64) for param in
                         initial_params]  # use int64 to avoid overflow

    # Aggregate each result scaled by the number of examples
    for _, fit_res in results:
        current_params = parameters_to_ndarrays(fit_res.parameters)
        for i, param in enumerate(current_params):
            scaled_param = param * fit_res.num_examples
            aggregated_params[i] += scaled_param

    final_params = [truncate_divide(param, num_examples_total) for param in aggregated_params]

    return final_params


class FedAvgInt(fl.server.strategy.FedAvg):
    def aggregate_fit(
            self,
            server_round: int,
            results: List[Tuple[ClientProxy, FitRes]],
            failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using weighted average with integer arithmetic."""
        if not results or (not self.accept_failures and failures):
            return None, {}

        assert not self.inplace, "Inplace aggregation is not supported for now."
        parameters_aggregated = aggregate(results)

        # Aggregate custom metrics if aggregation function was provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:  # Log warning once if no aggregation function
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        parameters_aggregated = ndarrays_to_parameters(parameters_aggregated)

        return parameters_aggregated, metrics_aggregated
