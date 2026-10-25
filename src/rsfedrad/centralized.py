"""Centralised baseline: the same CNN-LSTM trained on the union of all clients.

Each client's data is encoded and split exactly as in the federated setting;
the training and hold-out portions are then pooled on a single node.

    python -m rsfedrad.centralized
"""
import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf

from rsfedrad import config
from rsfedrad.data import load_client
from rsfedrad.metrics import binary_metrics
from rsfedrad.model import model_from_config


def main():
    parser = argparse.ArgumentParser(description="Train the centralised baseline")
    parser.add_argument("--out-dir", default=str(config.ROOT / "runs" / "centralized"))
    parser.add_argument("--config", default=str(config.CONFIG_PATH))
    args = parser.parse_args()

    cfg = config.load_config(args.config)
    d = cfg["data"]
    tf.keras.utils.set_random_seed(d["seed"])
    clients = [load_client(i, d["num_clients"], d["clients_dir"], d["test_size"], d["seed"])
               for i in range(1, d["num_clients"] + 1)]
    X_train = np.concatenate([c.X_train for c in clients])
    y_train = np.concatenate([c.y_train for c in clients])
    X_test = np.concatenate([c.X_test for c in clients])
    y_test = np.concatenate([c.y_test for c in clients])

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model = model_from_config(cfg)
    model.fit(
        X_train, y_train,
        epochs=cfg["centralized"]["epochs"],
        batch_size=cfg["centralized"]["batch_size"],
        callbacks=[tf.keras.callbacks.CSVLogger(str(out_dir / "training_history_centralized.csv"))],
    )
    metrics = binary_metrics(y_test, model.predict(X_test, verbose=0))
    model.save_weights(str(out_dir / "centralized_model.weights.h5"))
    (out_dir / "centralized_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
