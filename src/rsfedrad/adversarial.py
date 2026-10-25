"""Gradient-based adversarial examples used for adversarial training.

Both attacks perturb the standardised input sequence in the direction that
increases the binary cross-entropy of the current model. Perturbed values are
kept inside the range observed in the client's training data.
"""
import numpy as np
import tensorflow as tf


def _loss_gradient(model, X, y):
    X = tf.convert_to_tensor(X, dtype=tf.float32)
    y = tf.reshape(tf.convert_to_tensor(y, dtype=tf.float32), (-1, 1))
    with tf.GradientTape() as tape:
        tape.watch(X)
        loss = tf.keras.losses.binary_crossentropy(y, model(X, training=False))
    return tape.gradient(loss, X)


def fgsm(model, X, y, epsilon=0.01, clip_min=None, clip_max=None):
    """Fast Gradient Sign Method (Goodfellow et al., 2015)."""
    clip_min = X.min() if clip_min is None else clip_min
    clip_max = X.max() if clip_max is None else clip_max
    X_adv = X + epsilon * np.sign(_loss_gradient(model, X, y).numpy())
    return np.clip(X_adv, clip_min, clip_max).astype(np.float32)


def bim(model, X, y, epsilon=0.01, alpha=0.005, iterations=10, clip_min=None, clip_max=None):
    """Basic Iterative Method (Kurakin et al., 2017): iterated FGSM steps of size
    ``alpha`` projected onto the epsilon-ball around the original input."""
    clip_min = X.min() if clip_min is None else clip_min
    clip_max = X.max() if clip_max is None else clip_max
    X_adv = X.astype(np.float32).copy()
    for _ in range(iterations):
        X_adv = X_adv + alpha * np.sign(_loss_gradient(model, X_adv, y).numpy())
        X_adv = np.clip(X_adv, X - epsilon, X + epsilon)
        X_adv = np.clip(X_adv, clip_min, clip_max)
    return X_adv.astype(np.float32)


def generate(model, X, y, attack, cfg, batch_size=512):
    """Craft adversarial copies of ``X`` in batches."""
    lo, hi = float(X.min()), float(X.max())
    out = []
    for start in range(0, len(X), batch_size):
        xb, yb = X[start:start + batch_size], y[start:start + batch_size]
        if attack == "fgsm":
            out.append(fgsm(model, xb, yb, cfg["epsilon"], lo, hi))
        elif attack == "bim":
            out.append(bim(model, xb, yb, cfg["epsilon"], cfg["bim_alpha"], cfg["bim_iterations"], lo, hi))
        else:
            raise ValueError(f"unknown attack: {attack}")
    return np.concatenate(out)
