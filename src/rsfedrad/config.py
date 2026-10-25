"""Paths and experiment configuration."""
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs" / "default.yaml"
RESULTS_DIR = ROOT / "results"
LOGS_DIR = RESULTS_DIR / "logs"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"
XAI_DIR = RESULTS_DIR / "explainability"
MODELS_DIR = ROOT / "models"

# Layout of the 530-position behavioural sequence
FEATURE_GROUPS = {"ApiCall": 500, "Dll": 10, "Mutex": 10, "TTP": 10}
CLASS_NAMES = ["Benign", "Ransomware"]


def feature_names():
    return [f"{group}_{i}" for group, n in FEATURE_GROUPS.items() for i in range(n)]


def group_of(feature):
    return feature.rsplit("_", 1)[0]


def load_config(path=CONFIG_PATH):
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["data"]["clients_dir"] = str(ROOT / cfg["data"]["clients_dir"])
    cfg["explainability"]["model_path"] = str(ROOT / cfg["explainability"]["model_path"])
    return cfg
