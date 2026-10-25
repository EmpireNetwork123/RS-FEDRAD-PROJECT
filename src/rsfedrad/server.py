"""Flower server for RS-FEDRAD.

    python -m rsfedrad.server                       # FedAvg, 10 rounds, 10 clients
    python -m rsfedrad.server --strategy fedprox
    python -m rsfedrad.server --num-clients 15

After every round the server aggregates the clients' metrics (weighted by the
number of local samples), appends them to ``global_metrics.csv`` and stores the
global model as ``global_model_round_<r>.h5``.
"""
import argparse
import csv
import logging
from pathlib import Path

import flwr as fl

from rsfedrad import config
from rsfedrad.model import model_from_config

STRATEGIES = {
    "fedavg": fl.server.strategy.FedAvg,
    "fedprox": fl.server.strategy.FedProx,
    "fedavgm": fl.server.strategy.FedAvgM,
    "fedmedian": fl.server.strategy.FedMedian,
}


def weighted_average(results):
    total = sum(n for n, _ in results)
    keys = results[0][1].keys() if results else []
    return {k: sum(n * m[k] for n, m in results) / total for k in keys}


def build_strategy(cfg, num_clients, out_dir):
    fed = cfg["federated"]
    name = fed["strategy"].lower()
    base = STRATEGIES[name]

    class RSFedradStrategy(base):
        """Selected aggregation rule plus per-round logging and checkpointing."""

        def aggregate_fit(self, server_round, results, failures):
            aggregated = super().aggregate_fit(server_round, results, failures)
            parameters, metrics = aggregated
            if parameters is not None:
                self._save_round(server_round, parameters, metrics)
            return aggregated

        def _save_round(self, server_round, parameters, metrics):
            model = model_from_config(cfg)
            model.set_weights(fl.common.parameters_to_ndarrays(parameters))
            model.save_weights(str(out_dir / f"global_model_round_{server_round}.weights.h5"))
            path = out_dir / "global_metrics.csv"
            new = not path.exists()
            with open(path, "a", newline="") as f:
                writer = csv.writer(f)
                if new:
                    writer.writerow(["round", *metrics])
                writer.writerow([server_round, *metrics.values()])
            logging.info("round %d: %s", server_round, metrics)

    def fit_config(server_round):
        return {
            "server_round": server_round,
            "local_epochs": fed["local_epochs"],
            "batch_size": fed["batch_size"],
        }

    kwargs = dict(
        min_fit_clients=num_clients,
        min_evaluate_clients=num_clients,
        min_available_clients=num_clients,
        on_fit_config_fn=fit_config,
        fit_metrics_aggregation_fn=weighted_average,
        evaluate_metrics_aggregation_fn=weighted_average,
    )
    if name == "fedprox":
        kwargs["proximal_mu"] = fed["proximal_mu"]
    if name == "fedavgm":
        kwargs["server_momentum"] = fed["server_momentum"]
        # FedAvgM applies momentum relative to an initial global model
        kwargs["initial_parameters"] = fl.common.ndarrays_to_parameters(model_from_config(cfg).get_weights())
    return RSFedradStrategy(**kwargs)


def main():
    parser = argparse.ArgumentParser(description="Start the RS-FEDRAD Flower server")
    parser.add_argument("--strategy", choices=sorted(STRATEGIES), default=None)
    parser.add_argument("--num-clients", type=int, default=None)
    parser.add_argument("--rounds", type=int, default=None)
    parser.add_argument("--server-address", default=None)
    parser.add_argument("--out-dir", default=str(config.ROOT / "runs" / "federated"))
    parser.add_argument("--config", default=str(config.CONFIG_PATH))
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    cfg = config.load_config(args.config)
    if args.strategy:
        cfg["federated"]["strategy"] = args.strategy
    num_clients = args.num_clients or cfg["data"]["num_clients"]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fl.server.start_server(
        server_address=args.server_address or cfg["federated"]["server_address"],
        config=fl.server.ServerConfig(num_rounds=args.rounds or cfg["federated"]["num_rounds"]),
        strategy=build_strategy(cfg, num_clients, out_dir),
    )


if __name__ == "__main__":
    main()
