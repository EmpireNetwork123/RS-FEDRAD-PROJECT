import numpy as np
import pandas as pd
import pytest

from rsfedrad import config
from rsfedrad.aggregate_logs import aggregate
from rsfedrad.data import load_client, load_raw_client
from rsfedrad.encoding import ClientEncoder


def test_feature_layout():
    names = config.feature_names()
    assert len(names) == 530
    assert names[0] == "ApiCall_0" and names[500] == "Dll_0"
    assert names[510] == "Mutex_0" and names[520] == "TTP_0"


def test_client_files():
    sizes = []
    for i in range(1, 11):
        X, y = load_raw_client(i)
        assert X.shape[1] == 530
        assert set(np.unique(y)) == {0, 1}
        sizes.append(len(y))
    assert sum(sizes) == 30150


def test_encoder_roundtrip():
    X = pd.DataFrame({"a": ["NtClose", None, "NtOpenFile", "NtClose"], "b": ["x", "y", "y", None]})
    enc = ClientEncoder()
    codes = enc.fit(X)
    assert codes.shape == (4, 2)
    assert enc.decode("a", codes[2, 0]) == "NtOpenFile"
    # empty positions take the most frequent value of the column
    assert enc.decode("a", codes[1, 0]) == "NtClose"
    Z = enc.codes_to_model_input(codes)
    assert np.allclose(Z.mean(axis=0), 0)


def test_client_split_is_stratified():
    c = load_client(1)
    assert len(c.y_test) == pytest.approx(0.2 * (len(c.y_train) + len(c.y_test)), abs=1)
    assert abs(c.y_train.mean() - c.y_test.mean()) < 0.01
    assert c.X_train.shape[1:] == (530, 1)


def test_rescaled_partition():
    c = load_client(3, num_clients=15)
    assert 1900 <= len(c.y_train) + len(c.y_test) <= 2100


def test_logs_reproduce_published_table():
    table, n = aggregate(config.LOGS_DIR / "federated")
    published = pd.read_csv(config.TABLES_DIR / "federated_rounds.csv")
    assert n == 10
    assert np.allclose(table[published.columns[1:]].values, published.iloc[:, 1:].values, atol=0.005)
