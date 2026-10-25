"""Run a complete federation on one machine.

The script starts the Flower server and one process per client, all talking
over local gRPC exactly as in a distributed deployment.

    python -m rsfedrad.simulate
    python -m rsfedrad.simulate --strategy fedmedian --num-clients 20
    python -m rsfedrad.simulate --attack bim
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

import yaml

from rsfedrad import config
from rsfedrad.server import STRATEGIES


def main():
    parser = argparse.ArgumentParser(description="Run an RS-FEDRAD federation locally")
    parser.add_argument("--strategy", choices=sorted(STRATEGIES), default=None)
    parser.add_argument("--num-clients", type=int, default=None)
    parser.add_argument("--rounds", type=int, default=None)
    parser.add_argument("--local-epochs", type=int, default=None)
    parser.add_argument("--attack", choices=["none", "fgsm", "bim"], default="none")
    parser.add_argument("--out-dir", default=str(config.ROOT / "runs" / "simulation"))
    parser.add_argument("--config", default=str(config.CONFIG_PATH))
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if args.strategy:
        cfg["federated"]["strategy"] = args.strategy
    if args.local_epochs:
        cfg["federated"]["local_epochs"] = args.local_epochs
    cfg["data"]["clients_dir"] = str(config.ROOT / cfg["data"]["clients_dir"])
    cfg["explainability"]["model_path"] = str(config.ROOT / cfg["explainability"]["model_path"])
    run_config = out_dir / "config.yaml"
    run_config.write_text(yaml.safe_dump(cfg, sort_keys=False))

    num_clients = args.num_clients or cfg["data"]["num_clients"]
    server_cmd = [sys.executable, "-m", "rsfedrad.server", "--num-clients", str(num_clients),
                  "--config", str(run_config), "--out-dir", str(out_dir)]
    if args.rounds:
        server_cmd += ["--rounds", str(args.rounds)]
    server = subprocess.Popen(server_cmd)
    time.sleep(10)

    clients = [
        subprocess.Popen([sys.executable, "-m", "rsfedrad.client", "--client-id", str(i),
                          "--num-clients", str(num_clients), "--attack", args.attack,
                          "--config", str(run_config), "--log-dir", str(out_dir)])
        for i in range(1, num_clients + 1)
    ]
    for p in clients:
        p.wait()
    sys.exit(server.wait())


if __name__ == "__main__":
    main()
