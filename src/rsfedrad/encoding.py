"""Encoding of the raw behavioural sequences into model inputs.

Every client encodes its own data locally, so no vocabulary or statistic ever
leaves a participant. The steps follow the RS-FEDRAD preprocessing:

1. empty positions are filled with the most frequent value of that position;
2. each position is label-encoded (API, DLL, mutex and TTP names become integers);
3. the integer codes are scaled to [0, 1] (min-max);
4. the client standardises the scaled features (z-score) before training.

The fitted encoder keeps the classes of every position, which lets us map any
encoded value back to the API call, DLL, mutex or ATT&CK technique it stands for.
"""
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler


class ClientEncoder:
    def __init__(self):
        self.columns = None
        self.imputer = SimpleImputer(strategy="most_frequent")
        self.classes_ = {}
        self.minmax = MinMaxScaler()
        self.standard = StandardScaler()

    def fit_codes(self, X):
        """Impute and label-encode a raw frame; returns integer codes."""
        self.columns = list(X.columns)
        filled = pd.DataFrame(self.imputer.fit_transform(X.astype(object)), columns=self.columns)
        codes = np.empty(filled.shape, dtype=np.int64)
        for j, col in enumerate(self.columns):
            encoder = LabelEncoder().fit(filled[col].astype(str))
            self.classes_[col] = encoder.classes_
            codes[:, j] = encoder.transform(filled[col].astype(str))
        return codes

    def fit(self, X):
        codes = self.fit_codes(X)
        scaled = self.minmax.fit_transform(codes)
        self.standard.fit(scaled)
        return codes

    def codes_to_model_input(self, codes):
        """Integer codes -> standardised features of shape (n, 530)."""
        scaled = self.minmax.transform(np.asarray(codes, dtype=float))
        return self.standard.transform(scaled)

    def decode(self, column, code):
        return str(self.classes_[column][int(code)])

    def decode_row(self, codes):
        return {col: self.decode(col, c) for col, c in zip(self.columns, codes)}
