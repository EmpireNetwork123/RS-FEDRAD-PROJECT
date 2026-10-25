"""Explainability of the RS-FEDRAD global model with LIME and SHAP.

We follow the protocol of XRan (Gulmez et al., 2024) and extend it to the
TTP-enhanced feature space:

* LIME explains individual decisions. Each of the 530 positions is treated as a
  categorical variable, so an explanation reads ``Dll_2 = ADVAPI32dll`` rather
  than an encoded number. We explain correctly classified ransomware and benign
  samples as well as the misclassified ones.
* SHAP (expected gradients) gives global explanations: (i) location-based, the
  importance of every position of the concatenated sequence; (ii) feature-based,
  the API calls, DLLs, mutexes and ATT&CK techniques that push the model towards
  either class; and (iii) the share of the decision carried by each feature group,
  which quantifies the contribution of the TTP features.

    python -m rsfedrad.explain
"""
import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import tensorflow as tf
from lime.lime_base import LimeBase
from sklearn.metrics import pairwise_distances

from rsfedrad import config
from rsfedrad.data import load_client
from rsfedrad.model import load_global_model

EMPTY = "<empty>"
GROUPS = ("ApiCall", "Dll", "Mutex", "TTP")
EMPTY_LABELS = {"ApiCall": "(no API call)", "Dll": "(no DLL)", "Mutex": "(no mutex)", "TTP": "(no technique)"}

# MITRE ATT&CK techniques present in the TTP-enhanced dataset
ATTACK_TECHNIQUES = {
    "T1003": "OS Credential Dumping",
    "T1012": "Query Registry",
    "T1021": "Remote Services",
    "T1027": "Obfuscated Files or Information",
    "T1055": "Process Injection",
    "T1071": "Application Layer Protocol",
    "T1073": "DLL Side-Loading",
    "T1082": "System Information Discovery",
    "T1083": "File and Directory Discovery",
    "T1098": "Account Manipulation",
    "T1105": "Ingress Tool Transfer",
    "T1106": "Native API",
    "T1134": "Access Token Manipulation",
}


# ---------------------------------------------------------------- utilities
def predict_proba(model, X):
    p = model.predict(X, verbose=0, batch_size=1024).ravel()
    return np.c_[1 - p, p]


def token(client, row, j):
    """Readable value of position j of a sample; empty positions are marked."""
    col = client.encoder.columns[j]
    if client.missing[row, j]:
        return EMPTY
    return client.encoder.decode(col, client.codes[row, j])


def pretty(name):
    short = {"ApiCall": "API", "Dll": "DLL", "Mutex": "Mutex", "TTP": "TTP"}
    return f"{short[config.group_of(name)]} {name.rsplit('_', 1)[1]}"


def save(fig, name, out_dir):
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


# -------------------------------------------------------------- evaluation
def collect_test_predictions(model, clients):
    rows = []
    for c in clients:
        prob = predict_proba(model, c.X_test)[:, 1]
        for k, (row, y, p) in enumerate(zip(c.idx_test, c.y_test, prob)):
            rows.append({"client": c.client_id, "row": int(row), "pos": k, "label": int(y), "prob": float(p)})
    df = pd.DataFrame(rows)
    df["pred"] = (df.prob > 0.5).astype(int)
    df["correct"] = df.pred == df.label
    return df


def select_cases(preds, seed, n_correct=2, n_wrong=2):
    rng = np.random.default_rng(seed)
    picks = []
    for label, name in ((1, "Ransomware"), (0, "Benign")):
        ok = preds[(preds.label == label) & preds.correct]
        for i in rng.choice(ok.index, size=min(n_correct, len(ok)), replace=False):
            picks.append((i, f"correct {name.lower()}"))
    for label, name in ((0, "benign predicted as ransomware"), (1, "ransomware predicted as benign")):
        wrong = preds[(preds.label == label) & ~preds.correct]
        for i in wrong.sort_values("prob", ascending=label == 1).index[:n_wrong]:
            picks.append((i, f"misclassified {name}"))
    return picks


# --------------------------------------------------------------------- LIME
def baseline_codes(encoder):
    """Code an empty position receives after imputation (the per-position mode)."""
    return np.array([np.searchsorted(encoder.classes_[col], str(stat))
                     for col, stat in zip(encoder.columns, encoder.imputer.statistics_)])


def lime_explain_sequence(predict_fn, codes, baseline, num_samples, num_features, seed):
    """LIME over the observed events of one behavioural sequence.

    The interpretable representation is a binary vector over the positions whose
    value differs from an empty slot. Perturbed sequences remove a random subset of
    these events (the position takes the value of an empty slot), as LIME does
    for words in a document. Each weight is the local effect of observing that
    API call, DLL, mutex or TTP at that position instead of nothing, expressed
    in log-odds of the ransomware class.
    """
    rng = np.random.default_rng(seed)
    active = np.where(codes != baseline)[0]
    d = len(active)
    Z = np.ones((num_samples, d), dtype=int)
    for k in range(1, num_samples):
        drop = rng.choice(d, size=rng.integers(1, d + 1), replace=False)
        Z[k, drop] = 0
    perturbed = np.tile(codes, (num_samples, 1))
    for k in range(num_samples):
        off = active[Z[k] == 0]
        perturbed[k, off] = baseline[off]
    # the sigmoid output saturates near 0 and 1, so the surrogate is fitted on the
    # log-odds of the ransomware class
    p = np.clip(predict_fn(perturbed)[:, 1], 1e-7, 1 - 1e-7)
    logit = np.log(p / (1 - p))
    targets = np.c_[-logit, logit]
    distances = pairwise_distances(Z, Z[:1], metric="cosine").ravel() * 100
    base = LimeBase(kernel_fn=lambda x: np.sqrt(np.exp(-(x ** 2) / 25 ** 2)), random_state=seed)
    _, weights, score, _ = base.explain_instance_with_data(
        Z, targets, distances, label=1, num_features=num_features, feature_selection="auto")
    return [(int(active[j]), float(w)) for j, w in weights], float(score)


def lime_explanations(model, clients, preds, cfg, out_dir):
    xcfg = cfg["explainability"]
    names = config.feature_names()
    by_id = {c.client_id: c for c in clients}
    records = []

    for case_no, (i, kind) in enumerate(select_cases(preds, cfg["data"]["seed"]), start=1):
        case = preds.loc[i]
        c = by_id[case.client]

        def fn(codes, c=c):
            X = c.encoder.codes_to_model_input(codes)[..., None]
            return predict_proba(model, X)

        weights, score = lime_explain_sequence(
            fn, c.codes[case.row], baseline_codes(c.encoder),
            xcfg["lime_num_samples"], xcfg["lime_num_features"], cfg["data"]["seed"] + case_no)
        weights = sorted(weights, key=lambda t: abs(t[1]), reverse=True)
        items = []
        for j, w in weights:
            items.append({"case": case_no, "kind": kind, "client": int(c.client_id),
                          "label": config.CLASS_NAMES[case.label], "p_ransomware": round(case.prob, 4),
                          "feature": names[j], "value": token(c, case.row, j), "weight": w,
                          "local_r2": round(score, 3)})
        records.extend(items)

    df = pd.DataFrame(records)
    df.to_csv(out_dir / "lime_explanations.csv", index=False)
    return df


# --------------------------------------------------------------------- SHAP
def logit_model(model):
    """Same network with a linear output unit (the pre-sigmoid logit)."""
    inputs = tf.keras.Input(shape=model.input_shape[1:])
    x = inputs
    for layer in model.layers[:-1]:
        x = layer(x)
    out = tf.keras.layers.Dense(1)(x)
    lm = tf.keras.Model(inputs, out)
    lm.layers[-1].set_weights(model.layers[-1].get_weights())
    return lm


def shap_attributions(model, clients, preds, cfg):
    xcfg = cfg["explainability"]
    rng = np.random.default_rng(cfg["data"]["seed"])
    background = np.concatenate([c.X_train for c in clients])
    background = background[rng.choice(len(background), xcfg["shap_background"], replace=False)]

    picks = []
    for label in (0, 1):
        pool = preds[preds.label == label].index
        picks.extend(rng.choice(pool, size=min(xcfg["shap_samples_per_class"], len(pool)), replace=False))
    sample = preds.loc[picks].reset_index(drop=True)
    by_id = {c.client_id: c for c in clients}
    X = np.stack([by_id[r.client].X_test[r.pos] for r in sample.itertuples()])

    # attributions are computed on the logit of the output unit, where they add up
    # to f(x) - E[f(background)] without the distortion of the saturated sigmoid
    lm = logit_model(model)
    explainer = shap.GradientExplainer(lm, background)
    values = explainer.shap_values(X, nsamples=xcfg["shap_nsamples"], rseed=cfg["data"]["seed"])
    values = np.asarray(values[0] if isinstance(values, list) else values).reshape(len(X), -1)
    gap = lm.predict(X, verbose=0).ravel() - lm.predict(background, verbose=0).mean()
    completeness = float(np.corrcoef(values.sum(axis=1), gap)[0, 1])
    return sample, values, completeness


def location_importance(values, sample, out_dir):
    names = config.feature_names()
    groups = np.array([config.group_of(n) for n in names])
    importance = np.abs(values).mean(axis=0)
    loc = pd.DataFrame({"feature": names, "group": groups, "mean_abs_shap": importance,
                        "mean_shap_ransomware": values[sample.label.values == 1].mean(axis=0),
                        "mean_shap_benign": values[sample.label.values == 0].mean(axis=0)})
    loc.to_csv(out_dir / "shap_location_importance.csv", index=False)
    return loc


def group_contribution(values, sample, out_dir):
    names = config.feature_names()
    groups = np.array([config.group_of(n) for n in names])
    rows = []
    for g in GROUPS:
        m = groups == g
        abs_total = np.abs(values[:, m]).sum(axis=1)
        rows.append({
            "group": g,
            "positions": int(m.sum()),
            "share_of_total_abs_shap": abs_total.sum() / np.abs(values).sum(),
            "mean_abs_shap_per_position": np.abs(values[:, m]).mean(),
            "mean_net_shap_ransomware": values[sample.label.values == 1][:, m].sum(axis=1).mean(),
            "mean_net_shap_benign": values[sample.label.values == 0][:, m].sum(axis=1).mean(),
        })
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "shap_group_contribution.csv", index=False)
    return df


def token_attributions(values, sample, clients, out_dir, min_count=20):
    """Average SHAP value of every observed value (API call, DLL, mutex, TTP)."""
    names = config.feature_names()
    by_id = {c.client_id: c for c in clients}
    rows = []
    for i, r in enumerate(sample.itertuples()):
        c = by_id[r.client]
        row = c.idx_test[r.pos]
        for j, name in enumerate(names):
            rows.append((config.group_of(name), token(c, row, j), values[i, j]))
    df = pd.DataFrame(rows, columns=["group", "value", "shap"])
    agg = (df.groupby(["group", "value"]).shap.agg(["mean", "count"]).reset_index()
           .rename(columns={"mean": "mean_shap"}))
    agg = agg[agg["count"] >= min_count].sort_values("mean_shap", ascending=False)
    agg.to_csv(out_dir / "shap_value_attributions.csv", index=False)

    ttp = agg[agg.group == "TTP"].copy()
    ttp["technique"] = ttp.value.map(lambda v: f"{v} {ATTACK_TECHNIQUES[v]}" if v in ATTACK_TECHNIQUES else v)
    ttp.to_csv(out_dir / "shap_ttp_techniques.csv", index=False)
    return agg


# ------------------------------------------------------------------ figures
def display_value(group, value, width=30):
    """Readable label of an API call, DLL, mutex or technique."""
    if value == EMPTY:
        return EMPTY_LABELS[group]
    if group == "Dll":
        value = re.sub(r"dll$", ".dll", value, flags=re.IGNORECASE)
    if group == "TTP" and value in ATTACK_TECHNIQUES:
        value = f"{value} {ATTACK_TECHNIQUES[value]}"
    elif group == "TTP":
        value = f"{value} (TTP tag)"
    return value if len(value) <= width else value[:width - 1] + "…"


def lime_figure(lime_df, out_dir, cases=(1, 3, 8, 6), num_features=6):
    """Local explanations in the layout of XRan (Figs. 7 and 8)."""
    support, contradict = "#2ca02c", "#d62728"
    fig, axes = plt.subplots(2, 2, figsize=(10, 5.6))
    for ax, case_no, letter in zip(axes.ravel(), cases, "abcd"):
        rows = lime_df[lime_df.case == case_no].head(num_features)[::-1]
        p = rows.p_ransomware.iloc[0]
        pred = "Ransomware" if p > 0.5 else "Benign"
        group = rows.feature.map(config.group_of)
        labels = [f"{display_value(g, v, 34)}  [{pretty(f)}]" for g, v, f in zip(group, rows.value, rows.feature)]
        supports = (rows.weight > 0) == (pred == "Ransomware")
        y = np.arange(len(rows))
        ax.barh(y, rows.weight, color=np.where(supports, support, contradict), height=0.6)
        ax.set_yticks(y, labels, fontsize=7.5)
        ax.axvline(0, color="#444444", lw=0.6)
        ax.tick_params(axis="x", labelsize=7.5)
        ax.set_title(f"Case {case_no}: {rows.label.iloc[0]} sample\nModel's prediction: {pred} [{p:.4f}]",
                     fontsize=8.5)
        ax.text(0.5, -0.2, f"({letter})", transform=ax.transAxes, ha="center", va="top", fontsize=9)
    handles = [plt.Rectangle((0, 0), 1, 1, color=support), plt.Rectangle((0, 0), 1, 1, color=contradict)]
    fig.legend(handles, ["Supports model's decision", "Contradicts model's decision"],
               loc="lower center", ncol=2, frameon=False, fontsize=8.5, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(h_pad=2.2, w_pad=2.0)
    save(fig, "fig_lime_local_explanations", out_dir)


def location_figure(values, out_dir, max_display=20):
    """Mean |SHAP| of the most influential positions (XRan, Fig. 11)."""
    names = [pretty(n) for n in config.feature_names()]
    shap.summary_plot(values, feature_names=names, plot_type="bar", max_display=max_display,
                      color="#ff9f1c", show=False, plot_size=(6.4, 5.2))
    fig = plt.gcf()
    fig.axes[0].tick_params(labelsize=8.5)
    fig.axes[0].set_xlabel("mean(|SHAP value|) (average impact on model output magnitude)", fontsize=9)
    save(fig, "fig_shap_location_based", out_dir)


def token_matrix(values, sample, clients):
    """SHAP values summed per observed API call, DLL, mutex and technique of each sample."""
    names = config.feature_names()
    groups = [config.group_of(n) for n in names]
    by_id = {c.client_id: c for c in clients}
    index, rows = {}, []
    for i, r in enumerate(sample.itertuples()):
        c = by_id[r.client]
        row = c.idx_test[r.pos]
        for j, g in enumerate(groups):
            k = index.setdefault((g, token(c, row, j)), len(index))
            rows.append((i, k, values[i, j]))
    M = np.zeros((len(sample), len(index)))
    for i, k, v in rows:
        M[i, k] += v
    labels = [display_value(g, v) for g, v in index]
    return M, labels


def feature_figure(values, sample, clients, out_dir, max_display=10):
    """Most representative features of each class (XRan, Fig. 13)."""
    M, labels = token_matrix(values, sample, clients)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    for ax, label, letter in zip(axes, (1, 0), "ab"):
        rows = M[sample.label.values == label]
        expl = shap.Explanation(values=rows.mean(axis=0), feature_names=labels)
        shap.plots.bar(expl, max_display=max_display, ax=ax, show=False)
        lo, hi = ax.get_xlim()
        ax.set_xlim(lo - 0.18 * (hi - lo), hi + 0.08 * (hi - lo))
        for t in ax.texts:
            t.set_fontsize(8)
        ax.set_xlabel("mean(SHAP value) on the logit", fontsize=9)
        ax.tick_params(labelsize=8.5)
        ax.text(0.5, -0.16, f"({letter}) {config.CLASS_NAMES[label]} class", transform=ax.transAxes,
                ha="center", va="top", fontsize=10)
    fig.tight_layout(w_pad=4)
    save(fig, "fig_shap_feature_based", out_dir)


def make_figures(out_dir, clients=None, cfg=None):
    lime_df = pd.read_csv(out_dir / "lime_explanations.csv")
    values = np.load(out_dir / "shap_values.npy")
    sample = pd.read_csv(out_dir / "shap_samples.csv")
    if clients is None:
        d = cfg["data"]
        clients = [load_client(i, d["num_clients"], d["clients_dir"], d["test_size"], d["seed"])
                   for i in range(1, d["num_clients"] + 1)]
    lime_figure(lime_df, out_dir)
    location_figure(values, out_dir)
    feature_figure(values, sample, clients, out_dir)


# --------------------------------------------------------------------- main
def main():
    parser = argparse.ArgumentParser(description="LIME and SHAP explanations of RS-FEDRAD")
    parser.add_argument("--model", default=None, help="path to global model weights")
    parser.add_argument("--out-dir", default=str(config.XAI_DIR))
    parser.add_argument("--config", default=str(config.CONFIG_PATH))
    parser.add_argument("--figures-only", action="store_true",
                        help="redraw the figures from the saved explanations")
    args = parser.parse_args()

    cfg = config.load_config(args.config)
    d = cfg["data"]
    tf.keras.utils.set_random_seed(d["seed"])
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    clients = [load_client(i, d["num_clients"], d["clients_dir"], d["test_size"], d["seed"])
               for i in range(1, d["num_clients"] + 1)]
    if args.figures_only:
        make_figures(out_dir, clients)
        return
    model = load_global_model(args.model or cfg["explainability"]["model_path"], cfg)

    preds = collect_test_predictions(model, clients)
    preds.to_csv(out_dir / "test_predictions.csv", index=False)
    summary = {
        "test_samples": int(len(preds)),
        "misclassified": int((~preds.correct).sum()),
        "false_positives": int(((preds.pred == 1) & (preds.label == 0)).sum()),
        "false_negatives": int(((preds.pred == 0) & (preds.label == 1)).sum()),
    }
    print(json.dumps(summary, indent=2))

    lime_df = lime_explanations(model, clients, preds, cfg, out_dir)
    sample, values, completeness = shap_attributions(model, clients, preds, cfg)
    np.save(out_dir / "shap_values.npy", values.astype(np.float32))
    sample.to_csv(out_dir / "shap_samples.csv", index=False)
    location_importance(values, sample, out_dir)
    groups = group_contribution(values, sample, out_dir)
    token_attributions(values, sample, clients, out_dir)

    summary["lime_cases"] = int(lime_df.case.nunique())
    summary["shap_samples"] = int(len(sample))
    summary["shap_completeness_correlation"] = round(completeness, 4)
    summary["group_share_of_abs_shap"] = dict(zip(groups.group, groups.share_of_total_abs_shap.round(4)))
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    make_figures(out_dir, clients)


if __name__ == "__main__":
    main()
