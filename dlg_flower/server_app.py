import torch
from flwr.common import Metrics, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
from flwr.common import Context

from dlg_flower.task import LeNet, get_parameters, weights_init


def fit_metrics_aggregation_fn(metrics: list[tuple[int, Metrics]]) -> Metrics:
    return {}


def evaluate_metrics_aggregation_fn(metrics: list[tuple[int, Metrics]]) -> Metrics:
    total = sum(n for n, _ in metrics)
    accuracy = sum(n * m["accuracy"] for n, m in metrics) / total if total > 0 else 0.0
    return {"accuracy": accuracy}


def server_fn(context: Context) -> ServerAppComponents:
    num_rounds = int(context.run_config.get("num-server-rounds", 1))

    torch.manual_seed(50)
    net = LeNet()
    net.apply(weights_init)

    initial_parameters = ndarrays_to_parameters(get_parameters(net))

    strategy = FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=2,
        min_evaluate_clients=2,
        min_available_clients=2,
        initial_parameters=initial_parameters,
        fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
        evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
    )

    return ServerAppComponents(
        strategy=strategy,
        config=ServerConfig(num_rounds=num_rounds),
    )


app = ServerApp(server_fn=server_fn)
