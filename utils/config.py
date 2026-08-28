"""
config.py
=========
Central configuration for the AD-HyFormer project.
All hyperparameters, paths, and constants live here so every other module
imports from a single source of truth instead of hardcoding values.
"""

import os
import random
import numpy as np
import torch
from dataclasses import dataclass, field
from typing import List


@dataclass
class Config:
    # ---------------- Paths ----------------
    project_root: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_root: str = os.path.join(project_root, "datasets")
    weights_dir: str = os.path.join(project_root, "weights")
    results_dir: str = os.path.join(project_root, "results")
    reports_dir: str = os.path.join(project_root, "reports")

    # ---------------- Classes ----------------
    class_names: List[str] = field(default_factory=lambda: ["CN", "MCI", "AD"])
    num_classes: int = 3

    # ---------------- Cohorts ----------------
    train_cohort: str = "ADNI"
    external_cohorts: List[str] = field(default_factory=lambda: ["AIBL", "OASIS"])

    # ---------------- Preprocessing ----------------
    image_size: int = 224
    n_slices_per_plane: int = 5          # sagittal / coronal / axial
    planes: List[str] = field(default_factory=lambda: ["sagittal", "coronal", "axial"])
    intensity_clip_percentiles: tuple = (0.5, 99.5)

    # ---------------- CNN Patch Embedding ----------------
    cnn_in_channels: int = 1
    cnn_channels: List[int] = field(default_factory=lambda: [32, 64, 128])
    patch_embed_dim: int = 256

    # ---------------- Vision Transformer ----------------
    vit_embed_dim: int = 256
    vit_depth: int = 6
    vit_heads: int = 8
    vit_mlp_ratio: float = 4.0
    vit_dropout: float = 0.1

    # ---------------- Slice Attention Fusion ----------------
    fusion_hidden_dim: int = 256

    # ---------------- Uncertainty ----------------
    mc_dropout_passes: int = 20
    evidential_lambda: float = 0.1        # KL regularization weight

    # ---------------- Training ----------------
    batch_size: int = 8
    epochs: int = 100
    lr: float = 3e-4
    weight_decay: float = 1e-4
    early_stopping_patience: int = 15
    mixed_precision: bool = True
    num_workers: int = 4
    seed: int = 42

    # ---------------- Device ----------------
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed: int = 42):
    """Fix all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


cfg = Config()

if __name__ == "__main__":
    set_seed(cfg.seed)
    print(cfg)
