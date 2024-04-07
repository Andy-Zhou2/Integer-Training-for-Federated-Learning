import numpy as np
import flwr as fl
from typing import Callable, Dict, List, Optional, Tuple, Union

from flwr.common import (
    FitRes,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_proxy import ClientProxy


class FedAvgInt(fl.server.strategy.FedAvg):
    def aggregate_fit(
            self,
            server_round: int,
            results: List[Tuple[ClientProxy, FitRes]],
            failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        parameters_aggregated, metrics_aggregated = super().aggregate_fit(server_round, results, failures)

        # Convert the parameters to integers
        params = parameters_to_ndarrays(parameters_aggregated)
        params = [np.rint(p).astype(np.int_) for p in params]
        parameters_aggregated = ndarrays_to_parameters(params)

        return parameters_aggregated, metrics_aggregated
