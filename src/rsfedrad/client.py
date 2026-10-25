"""Flower client: one participating organisation.

Usage (one terminal per client, after starting the server):

    python -m rsfedrad.client --client-id 1
    python -m rsfedrad.client --client-id 2 --attack fgsm   # adversarial training
"""
import argparse
import csv
from pathlib import Path

import flwr as fl
import numpy as np
import tensorflow as tf

from rsfedrad import adversarial, config
from rsfedrad.data import load_client
from rsfedrad.metrics import binary_metrics
from rsfedrad.model import compile_model, model_from_config


class RansomwareClient(fl.client.NumPyClient):
    def __init__(self, data, cfg, attack="none", log_dir=None):
        self.data = data
        self.cfg = cfg
        self.attack = attack
        self.name = f"client_{data.client_id:02d}"
        self.log_dir = Path(log_dir) if log_dir else None
        self.model = model_from_config(cfg)

    def get_parameters(self, config):
        return self.model.get_weights()

    def _training_set(self):
        X, y = self.data.X_train, self.data.y_train
        if self.attack == "none":
            return X, y
        # adversarial training: original samples plus adversarial copies crafted
        # against the current global model
        X_adv = adversarial.generate(self.model, X, y, self.attack, self.cfg["adversarial"])
        return np.concatenate([X, X_adv]), np.concatenate([y, y])

    def _fit_proximal(self, X, y, global_weights, mu, epochs, batch_size):
        """FedProx local objective: loss + mu/2 * ||w - w_global||^2."""
        optimizer = self.model.optimizer
        bce = tf.keras.losses.BinaryCrossentropy()
        anchors = [tf.constant(w) for w in global_weights]
        dataset = tf.data.Dataset.from_tensor_slices((X.astype("float32"), y.astype("float32")))
        history = {"accuracy": [], "loss": []}
        for _ in range(epochs):
            acc = tf.keras.metrics.BinaryAccuracy()
            mean_loss = tf.keras.metrics.Mean()
            for xb, yb in dataset.shuffle(len(X)).batch(batch_size):
                with tf.GradientTape() as tape:
                    pred = self.model(xb, training=True)
                    loss = bce(yb[:, None], pred)
                    prox = tf.add_n([tf.reduce_sum(tf.square(w - a))
                                     for w, a in zip(self.model.trainable_weights, anchors)])
                    total = loss + 0.5 * mu * prox
                grads = tape.gradient(total, self.model.trainable_weights)
                optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
                acc.update_state(yb[:, None], pred)
                mean_loss.update_state(loss)
            history["accuracy"].append(float(acc.result()))
            history["loss"].append(float(mean_loss.result()))
        return history

    def fit(self, parameters, config):
        self.model.set_weights(parameters)
        compile_model(self.model, self.cfg["model"]["learning_rate"])
        fed = self.cfg["federated"]
        epochs = int(config.get("local_epochs", fed["local_epochs"]))
        batch_size = int(config.get("batch_size", fed["batch_size"]))
        X, y = self._training_set()

        mu = float(config.get("proximal_mu", 0.0))
        if mu > 0:
            # trainable weights of the received global model (FedProx anchor)
            history = self._fit_proximal(X, y, [w.numpy() for w in self.model.trainable_weights],
                                         mu, epochs, batch_size)
        else:
            callbacks = []
            if self.log_dir:
                callbacks.append(tf.keras.callbacks.CSVLogger(
                    str(self.log_dir / f"training_history_{self.name}.csv"), append=True))
            history = self.model.fit(X, y, epochs=epochs, batch_size=batch_size,
                                     verbose=0, callbacks=callbacks).history

        test = binary_metrics(self.data.y_test, self.model.predict(self.data.X_test, verbose=0))
        metrics = {
            "train_accuracy": float(history["accuracy"][-1]),
            "train_loss": float(history["loss"][-1]),
            "test_accuracy": test["accuracy"],
            "precision": test["precision"],
            "recall": test["recall"],
            "f1_score": test["f1_score"],
        }
        self._log_round(int(config.get("server_round", 0)), metrics)
        return self.model.get_weights(), len(X), metrics

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        compile_model(self.model, self.cfg["model"]["learning_rate"])
        loss, _ = self.model.evaluate(self.data.X_test, self.data.y_test, verbose=0)
        metrics = binary_metrics(self.data.y_test, self.model.predict(self.data.X_test, verbose=0))
        return float(loss), len(self.data.y_test), metrics

    def _log_round(self, server_round, metrics):
        if not self.log_dir:
            return
        path = self.log_dir / f"{self.name}_metrics.csv"
        new = not path.exists()
        with open(path, "a", newline="") as f:
            writer = csv.writer(f)
            if new:
                writer.writerow(["round", *metrics])
            writer.writerow([server_round, *metrics.values()])


def make_client(client_id, cfg, attack="none", num_clients=None, log_dir=None):
    d = cfg["data"]
    data = load_client(client_id, num_clients or d["num_clients"], d["clients_dir"], d["test_size"], d["seed"])
    if log_dir:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
    return RansomwareClient(data, cfg, attack, log_dir)


def main():
    parser = argparse.ArgumentParser(description="Start one RS-FEDRAD client")
    parser.add_argument("--client-id", type=int, required=True)
    parser.add_argument("--num-clients", type=int, default=None)
    parser.add_argument("--attack", choices=["none", "fgsm", "bim"], default="none")
    parser.add_argument("--server-address", default=None)
    parser.add_argument("--log-dir", default=str(config.ROOT / "runs" / "federated"))
    parser.add_argument("--config", default=str(config.CONFIG_PATH))
    args = parser.parse_args()

    cfg = config.load_config(args.config)
    tf.keras.utils.set_random_seed(cfg["data"]["seed"] + args.client_id)
    client = make_client(args.client_id, cfg, args.attack, args.num_clients, args.log_dir)
    fl.client.start_client(
        server_address=args.server_address or cfg["federated"]["server_address"],
        client=client.to_client(),
    )


if __name__ == "__main__":
    main()
