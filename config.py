"""
config.py — YAML-based experiment configuration for MTP-GO

Usage:
    from config import load_config
    cfg = load_config("configs/my_exp.yaml")

The config namespace is compatible with the existing argparse-based interface.
Missing keys fall back to defaults listed in DEFAULTS below.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict

import yaml

# ─────────────────────────────────────────────────────────────────────────────
# Defaults  (mirrors argument_parser.py)
# ─────────────────────────────────────────────────────────────────────────────
DEFAULTS: Dict[str, Any] = {
    # ── Trainer ──────────────────────────────────────────────────────────────
    "epochs":           100,
    "batch_size":       1024,
    "lr":               1e-4,
    "clip":             5.0,
    "teacher_forcing":  0.2,
    "log_interval":     100,

    # ── Model ─────────────────────────────────────────────────────────────────
    "u1_lim":           10.0,
    "u2_lim":           10.0,
    "ode_solver":       "rk4",
    "n_mixtures":       8,
    "motion_model":     "2Xnode",
    "hidden_size":      64,
    "gnn_layer":        "natt",
    "n_gnn_layers":     1,
    "n_ode_hidden":     16,
    "n_ode_layers":     1,
    "n_heads":          1,
    "init_static":      False,
    "n_ode_static":     False,
    "use_edge_features": True,

    # ── Data / Program ────────────────────────────────────────────────────────
    "dataset":          "highD",
    "sparse":           False,
    "seed":             42,
    "use_logger":       False,
    "use_cuda":         True,
    "n_workers":        1,
    "store_data":       True,
    "overwrite_data":   False,
    "ckpt_dir":         "ckpts",
    "add_name":         "",
    "dry_run":          False,
    "tune_lr":          False,
    "tune_batch_size":  False,
    "small_ds":         False,
    "use_importance":   True,   # highD-imp only: False = drop importance → 6D features

    # ── Preprocessing (preprocess.py) ─────────────────────────────────────────
    "slot_importance_alpha": 0.0,   # >0 boosts I by empirical slot weight
    "gate_topn":             0,     # keep only top-N slots per timestep (0 = all)
    "gate_mask":             False, # NaN-fill gated slots (treat as absent)
}


# ─────────────────────────────────────────────────────────────────────────────
# Loader
# ─────────────────────────────────────────────────────────────────────────────

def load_config(config_path: str | Path) -> SimpleNamespace:
    """
    Load a YAML config file and return a SimpleNamespace.

    Unknown keys in the YAML are accepted (forward-compatible).
    Missing keys fall back to DEFAULTS.

    The namespace also provides two extra attributes:
        cfg.config_path  — resolved Path to the YAML file
        cfg.config_name  — stem of the YAML filename  (e.g. "test1")
    """
    path = Path(config_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open() as f:
        user_cfg: Dict[str, Any] = yaml.safe_load(f) or {}

    # Merge: defaults first, user values override
    merged = {**DEFAULTS, **user_cfg}

    ns = SimpleNamespace(**merged)
    ns.config_path = path
    ns.config_name = path.stem
    return ns


def parse_config() -> SimpleNamespace:
    """
    Parse --config <path> from sys.argv and return the loaded config.
    Called at module level in train.py / test.py.
    """
    parser = argparse.ArgumentParser(
        description="MTP-GO — load experiment config from YAML",
        add_help=True,
    )
    parser.add_argument(
        "--config", required=True, metavar="PATH",
        help="Path to YAML experiment config  (e.g. configs/test1.yaml)",
    )
    parser.add_argument(
        "--ckpt_dir", default=None, metavar="DIR",
        help="Root directory for checkpoints (overrides YAML ckpt_dir)",
    )
    # Allow unknown args so this can coexist with Lightning's own CLI parsing
    args, _ = parser.parse_known_args()
    ns = load_config(args.config)
    if args.ckpt_dir is not None:
        ns.ckpt_dir = args.ckpt_dir
    return ns
