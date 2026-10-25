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
GROUP_COLORS = {"ApiCall": "#4C72B0", "Dll": "#DD8452", "Mutex": "#55A868", "TTP": "#C44E52"}
GROUP_LABELS = {"ApiCall": "API calls", "Dll": "DLLs", "Mutex": "Mutexes", "TTP": "TTPs"}
RANSOM_COLOR, BENIGN_COLOR = "#C44E52", "#4C72B0"

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
    fig.savefig(out_dir / name, dpi=200, bbox_inches="tight")
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
        plot_lime_case(items, case_no, kind, case, out_dir)

    df = pd.DataFrame(records)
    df.to_csv(out_dir / "lime_explanations.csv", index=False)
    return df


def plot_lime_case(items, case_no, kind, case, out_dir):
    items = items[::-1]
    labels = [f"{pretty(it['feature'])} = {shorten(it['value'], 40)}" for it in items]
    w = [it["weight"] for it in items]
    fig, ax = plt.subplots(figsize=(7.5, 0.42 * len(items) + 1.2))
    ax.barh(np.arange(len(w)), w, color=[RANSOM_COLOR if v > 0 else BENIGN_COLOR for v in w])
    ax.set_yticks(np.arange(len(w)), labels)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("LIME weight on log-odds  (< 0 benign  |  ransomware > 0)")
    ax.set_title(f"Case {case_no}: {kind}\ntrue class {config.CLASS_NAMES[case.label]}, "
                 f"P(ransomware) = {case.prob:.3f}", fontsize=10)
    ax.tick_params(axis="y", labelsize=8)
    save(fig, f"lime_case_{case_no}.png", out_dir)


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

    fig, ax = plt.subplots(figsize=(11, 3.6))
    for g, color in GROUP_COLORS.items():
        m = groups == g
        ax.bar(np.where(m)[0], importance[m], color=color, width=1.0, label=GROUP_LABELS[g])
    ax.set_xlim(-2, len(names) + 1)
    ax.set_xlabel("Position in the concatenated sequence")
    ax.set_ylabel("mean |SHAP value| (logit)")
    ax.set_title("Location-based global explanation")
    ax.legend(ncol=4, fontsize=8, frameon=False)
    save(fig, "shap_location_importance.png", out_dir)

    top = loc.sort_values("mean_abs_shap", ascending=False).head(20)[::-1]
    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.barh([pretty(f) for f in top.feature], top.mean_abs_shap, color=[GROUP_COLORS[g] for g in top.group])
    ax.set_xlabel("mean |SHAP value|")
    ax.set_title("Twenty most influential positions")
    save(fig, "shap_top_positions.png", out_dir)
    return loc


def group_contribution(values, sample, out_dir):
    names = config.feature_names()
    groups = np.array([config.group_of(n) for n in names])
    rows = []
    for g in GROUP_COLORS:
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

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    labels = [GROUP_LABELS[g] for g in df.group]
    colors = [GROUP_COLORS[g] for g in df.group]
    axes[0].bar(labels, df.share_of_total_abs_shap * 100, color=colors)
    axes[0].set_ylabel("share of total |SHAP| (%)")
    axes[0].set_title("Share of the decision per feature group")
    axes[1].bar(labels, df.mean_abs_shap_per_position, color=colors)
    axes[1].set_ylabel("mean |SHAP| per position")
    axes[1].set_title("Influence of a single position")
    save(fig, "shap_group_contribution.png", out_dir)
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

    plot_value_attributions(agg, out_dir)
    return agg


def shorten(text, width=34):
    return text if len(text) <= width else f"{text[:width - 12]}...{text[-9:]}"


def plot_value_attributions(agg, out_dir):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    for ax, g in zip(axes, ["ApiCall", "Dll", "Mutex"]):
        sub = agg[(agg.group == g) & (agg.value != EMPTY)]
        top = pd.concat([sub.head(8), sub.tail(8)]).drop_duplicates()[::-1]
        y = np.arange(len(top))
        ax.barh(y, top.mean_shap, color=[RANSOM_COLOR if v > 0 else BENIGN_COLOR for v in top.mean_shap])
        ax.set_yticks(y, [shorten(v) for v in top.value], fontsize=8)
        ax.axvline(0, color="black", lw=0.8)
        ax.set_title(GROUP_LABELS[g])
        ax.set_xlabel("mean SHAP value (logit)")
    fig.suptitle("Feature-based global explanation (> 0 ransomware, < 0 benign)", y=1.01)
    fig.tight_layout()
    save(fig, "shap_value_attributions.png", out_dir)

    ttp = agg[agg.group == "TTP"].copy()
    ttp["technique"] = ttp.value.map(lambda v: f"{v} {ATTACK_TECHNIQUES[v]}" if v in ATTACK_TECHNIQUES else v)
    ttp.to_csv(out_dir / "shap_ttp_techniques.csv", index=False)
    ttp = ttp[::-1]
    fig, ax = plt.subplots(figsize=(7.5, 0.38 * len(ttp) + 1.2))
    ax.barh(ttp.technique, ttp.mean_shap,
            color=[RANSOM_COLOR if v > 0 else BENIGN_COLOR for v in ttp.mean_shap])
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("mean SHAP value on logit  (< 0 benign  |  ransomware > 0)")
    ax.set_title("Contribution of MITRE ATT&CK techniques")
    save(fig, "shap_ttp_techniques.png", out_dir)


# --------------------------------------------------------------------- main
def main():
    parser = argparse.ArgumentParser(description="LIME and SHAP explanations of RS-FEDRAD")
    parser.add_argument("--model", default=None, help="path to global model weights")
    parser.add_argument("--out-dir", default=str(config.XAI_DIR))
    parser.add_argument("--config", default=str(config.CONFIG_PATH))
    args = parser.parse_args()

    cfg = config.load_config(args.config)
    d = cfg["data"]
    tf.keras.utils.set_random_seed(d["seed"])
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    clients = [load_client(i, d["num_clients"], d["clients_dir"], d["test_size"], d["seed"])
               for i in range(1, d["num_clients"] + 1)]
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


if __name__ == "__main__":
    main()
