"""Loading and splitting of the client datasets."""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

from rsfedrad import config
from rsfedrad.encoding import ClientEncoder


def client_path(client_id, clients_dir=None):
    clients_dir = Path(clients_dir or config.ROOT / "data" / "clients")
    return clients_dir / f"client_{int(client_id):02d}.csv.gz"


def load_raw_client(client_id, clients_dir=None):
    df = pd.read_csv(client_path(client_id, clients_dir), low_memory=False, keep_default_na=True)
    X = df[config.feature_names()]
    y = df["label"].astype(int).to_numpy()
    return X, y


def load_all_clients(num_clients=10, clients_dir=None):
    frames = [load_raw_client(i, clients_dir) for i in range(1, num_clients + 1)]
    X = pd.concat([f[0] for f in frames], ignore_index=True)
    y = np.concatenate([f[1] for f in frames])
    return X, y


@dataclass
class ClientData:
    client_id: int
    encoder: ClientEncoder
    codes: np.ndarray            # integer codes of every sample, (n, 530)
    missing: np.ndarray          # True where the raw position was empty (filled by imputation)
    X_train: np.ndarray          # standardised, (n_train, 530, 1)
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    idx_train: np.ndarray        # row indices into ``codes``
    idx_test: np.ndarray


def prepare_client(X_raw, y, client_id=0, test_size=0.2, seed=42):
    """Encode one client's data and create its stratified 80/20 split."""
    encoder = ClientEncoder()
    codes = encoder.fit(X_raw)
    X = encoder.codes_to_model_input(codes)
    idx = np.arange(len(y))
    idx_train, idx_test = train_test_split(
        idx, test_size=test_size, stratify=y, shuffle=True, random_state=seed
    )
    return ClientData(
        client_id=client_id,
        encoder=encoder,
        codes=codes,
        missing=X_raw.isna().to_numpy(),
        X_train=X[idx_train][..., None],
        X_test=X[idx_test][..., None],
        y_train=y[idx_train],
        y_test=y[idx_test],
        idx_train=idx_train,
        idx_test=idx_test,
    )


def load_client(client_id, num_clients=10, clients_dir=None, test_size=0.2, seed=42):
    """Return the prepared data of one client.

    With the default 10 clients, each client reads its own file. For scalability
    experiments with more clients, the pooled dataset is re-partitioned into
    ``num_clients`` stratified shards and the requested shard is returned.
    """
    if num_clients == 10:
        X_raw, y = load_raw_client(client_id, clients_dir)
    else:
        X_all, y_all = load_all_clients(10, clients_dir)
        folds = StratifiedKFold(n_splits=num_clients, shuffle=True, random_state=seed)
        shard = list(folds.split(X_all, y_all))[client_id - 1][1]
        X_raw, y = X_all.iloc[shard].reset_index(drop=True), y_all[shard]
    return prepare_client(X_raw, y, client_id, test_size, seed)
