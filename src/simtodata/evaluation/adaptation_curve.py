"""Adaptation efficiency: fine-tune sample count vs performance."""

import copy

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from simtodata.data.dataset import InspectionDataset
from simtodata.data.transforms import Normalize
from simtodata.evaluation.metrics import compute_macro_f1
from simtodata.models.predict import predict_batch
from simtodata.models.train import train_model


def run_adaptation_sweep(model_template, pretrained_path, adapt_dataset, eval_loader,
                         sample_counts=(0, 25, 50, 100, 200), n_repeats=3,
                         ft_epochs=20, ft_lr=1e-4):
    """Run adaptation efficiency sweep.

    Args:
        model_template: Uninitialized model (for architecture).
        pretrained_path: Path to pretrained weights.
        adapt_dataset: Full adaptation InspectionDataset.
        eval_loader: DataLoader for evaluation.
        sample_counts: List of fine-tune sample counts to try. Counts exceeding
                       the adapt dataset size are skipped (not silently truncated).
        n_repeats: Repeats per count for error bars.
        ft_epochs: Fine-tuning epochs.
        ft_lr: Fine-tuning learning rate.

    Returns:
        Dict mapping count -> {'mean_f1': float, 'std_f1': float}.
    """
    adapt_size = len(adapt_dataset)
    results = {}
    for count in sample_counts:
        if count > adapt_size:
            print(f"  Skipping {count} samples (adapt dataset has only {adapt_size})")
            continue
        f1_scores = []
        for repeat in range(n_repeats):
            model = copy.deepcopy(model_template)
            model.load_state_dict(torch.load(pretrained_path, weights_only=True))

            if count == 0:
                # No fine-tuning, just evaluate
                preds, _, labels = predict_batch(model, eval_loader)
                f1_scores.append(compute_macro_f1(labels, preds))
                break  # No variance for 0 samples
            else:
                rng = np.random.default_rng(42 + repeat)
                indices = rng.choice(adapt_size, size=count, replace=False)
                subset = Subset(adapt_dataset, indices.tolist())
                ft_loader = DataLoader(subset, batch_size=min(64, count), shuffle=True)
                model, _ = train_model(model, ft_loader, epochs=ft_epochs, lr=ft_lr)
                preds, _, labels = predict_batch(model, eval_loader)
                f1_scores.append(compute_macro_f1(labels, preds))

        results[count] = {
            "mean_f1": float(np.mean(f1_scores)),
            "std_f1": float(np.std(f1_scores)) if len(f1_scores) > 1 else 0.0,
        }
        print(f"  {count} samples: F1 = {results[count]['mean_f1']:.4f} +/- {results[count]['std_f1']:.4f}")

    return results
