"""Hybrid CNN-LSTM detector (Algorithm 1 of the paper)."""
import tensorflow as tf


def create_cnn_lstm_model(
    sequence_length=530,
    conv_filters=64,
    kernel_size=3,
    pool_size=2,
    lstm_units=100,
    dense_units=64,
    dropout=0.5,
    learning_rate=0.001,
):
    model = tf.keras.Sequential(
        [
            tf.keras.Input(shape=(sequence_length, 1)),
            # spatial patterns within neighbouring behavioural events
            tf.keras.layers.Conv1D(conv_filters, kernel_size=kernel_size, activation="relu"),
            tf.keras.layers.MaxPooling1D(pool_size=pool_size),
            # temporal dependencies along the execution sequence
            tf.keras.layers.LSTM(lstm_units),
            tf.keras.layers.Dense(dense_units, activation="relu"),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ],
        name="rsfedrad_cnn_lstm",
    )
    compile_model(model, learning_rate)
    return model


def compile_model(model, learning_rate=0.001):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )


def model_from_config(cfg):
    return create_cnn_lstm_model(**cfg["model"])


def load_global_model(path, cfg=None):
    """Load saved global weights into a freshly built network."""
    model = model_from_config(cfg) if cfg else create_cnn_lstm_model()
    model.load_weights(path)
    return model
