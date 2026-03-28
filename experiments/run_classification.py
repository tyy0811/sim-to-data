"""Run classification benchmark experiments B1-B5."""

import argparse
import json
import os

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from simtodata.data.dataset import InspectionDataset
from simtodata.data.transforms import Normalize
from simtodata.models.factory import model_from_config
from simtodata.models.predict import predict_batch
from simtodata.models.train import train_model
from simtodata.evaluation.metrics import compute_all_metrics


def _save_result(name, metrics, results_dir, y_true=None, y_pred=None, y_proba=None):
    os.makedirs(results_dir, exist_ok=True)
    result = {"name": name, "metrics": metrics}
    if y_true is not None:
        result["y_true"] = y_true.tolist()
    if y_pred is not None:
        result["y_pred"] = y_pred.tolist()
    if y_proba is not None:
        result["y_proba"] = y_proba.tolist()
    path = os.path.join(results_dir, f"{name}.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Saved: {path}")


def _seed_everything(seed: int):
    """Set all random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _seeded_loader(dataset, batch_size, shuffle, seed):
    """Create a DataLoader with a seeded generator for reproducible shuffling."""
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=generator)


def run_all_experiments(seed=42, results_dir="results", models_dir="models",
                        data_dir="data", config_path="configs/model_cnn1d.yaml",
                        quick=False, device="cpu"):
    """Run B1-B5 experiments for a single training seed on fixed datasets.

    Parameters
    ----------
    seed : int
        Controls model init, DataLoader shuffling, and dropout masks.
    results_dir : str
        Directory for result JSONs (with y_true/y_pred/y_proba).
    models_dir : str
        Directory for intermediate model checkpoints.
    data_dir : str
        Directory containing the fixed .npz datasets.
    config_path : str
        Path to model YAML config.
    quick : bool
        If True, reduce epochs for fast iteration.
    device : str
        Torch device ("cpu" or "cuda").
    """
    _seed_everything(seed)

    with open(config_path) as f:
        config = yaml.safe_load(f)
    tc = config["training"]
    ft = config["finetune"]
    if quick:
        tc = {**tc, "epochs": 5, "early_stopping_patience": 3}
        ft = {**ft, "epochs": 3}
    os.makedirs(models_dir, exist_ok=True)

    norm = Normalize()
    source_train = InspectionDataset(f"{data_dir}/source_train.npz", transform=norm)
    source_val = InspectionDataset(f"{data_dir}/source_val.npz", transform=norm)
    source_test = InspectionDataset(f"{data_dir}/source_test.npz", transform=norm)
    shifted_test = InspectionDataset(f"{data_dir}/shifted_test.npz", transform=norm)
    adapt_data = InspectionDataset(f"{data_dir}/shifted_adapt.npz", transform=norm)
    randomized_train = InspectionDataset(f"{data_dir}/randomized_train.npz", transform=norm)

    bs = tc["batch_size"]
    train_loader = _seeded_loader(source_train, bs, shuffle=True, seed=seed)
    val_loader = _seeded_loader(source_val, bs, shuffle=False, seed=seed)
    source_test_loader = _seeded_loader(source_test, bs, shuffle=False, seed=seed)
    shifted_test_loader = _seeded_loader(shifted_test, bs, shuffle=False, seed=seed)
    adapt_loader = _seeded_loader(adapt_data, bs, shuffle=True, seed=seed)
    rand_train_loader = _seeded_loader(randomized_train, bs, shuffle=True, seed=seed)

    # B1: Source -> Source
    print("B1: Training on source, eval on source...")
    model_b1 = model_from_config(config_path)
    model_b1, _ = train_model(model_b1, train_loader, val_loader, epochs=tc["epochs"],
                              lr=tc["learning_rate"], weight_decay=tc["weight_decay"],
                              patience=tc["early_stopping_patience"],
                              scheduler_patience=tc["scheduler_patience"],
                              scheduler_factor=tc["scheduler_factor"], device=device)
    torch.save(model_b1.cpu().state_dict(), f"{models_dir}/B1_cnn1d_source.pt")
    preds, probs, labels = predict_batch(model_b1, source_test_loader, device=device)
    metrics = compute_all_metrics(labels, preds, probs)
    print(f"  B1 Macro-F1: {metrics['macro_f1']:.4f}")
    _save_result("B1_cnn1d_source_on_source", metrics, results_dir, labels, preds, probs)

    # B2: Source -> Shifted (reuse B1 model)
    print("B2: Eval B1 on shifted...")
    preds, probs, labels = predict_batch(model_b1, shifted_test_loader, device=device)
    metrics = compute_all_metrics(labels, preds, probs)
    print(f"  B2 Macro-F1: {metrics['macro_f1']:.4f}")
    _save_result("B2_cnn1d_source_on_shifted", metrics, results_dir, labels, preds, probs)

    # B3: Randomized -> Shifted
    print("B3: Training on randomized, eval on shifted...")
    model_b3 = model_from_config(config_path)
    model_b3, _ = train_model(model_b3, rand_train_loader, val_loader, epochs=tc["epochs"],
                              lr=tc["learning_rate"], weight_decay=tc["weight_decay"],
                              patience=tc["early_stopping_patience"],
                              scheduler_patience=tc["scheduler_patience"],
                              scheduler_factor=tc["scheduler_factor"], device=device)
    torch.save(model_b3.cpu().state_dict(), f"{models_dir}/B3_cnn1d_randomized.pt")
    preds, probs, labels = predict_batch(model_b3, shifted_test_loader, device=device)
    metrics = compute_all_metrics(labels, preds, probs)
    print(f"  B3 Macro-F1: {metrics['macro_f1']:.4f}")
    _save_result("B3_cnn1d_randomized_on_shifted", metrics, results_dir, labels, preds, probs)

    # B4: Source + fine-tune -> Shifted
    print("B4: Fine-tuning B1 on adapt, eval on shifted...")
    model_b4 = model_from_config(config_path)
    model_b4.load_state_dict(torch.load(f"{models_dir}/B1_cnn1d_source.pt", weights_only=True))
    model_b4, _ = train_model(model_b4, adapt_loader, epochs=ft["epochs"],
                              lr=ft["learning_rate"], device=device)
    torch.save(model_b4.cpu().state_dict(), f"{models_dir}/B4_cnn1d_source_finetuned.pt")
    preds, probs, labels = predict_batch(model_b4, shifted_test_loader, device=device)
    metrics = compute_all_metrics(labels, preds, probs)
    print(f"  B4 Macro-F1: {metrics['macro_f1']:.4f}")
    _save_result("B4_cnn1d_source_finetune_on_shifted", metrics, results_dir, labels, preds, probs)

    # B5: Randomized + fine-tune -> Shifted
    print("B5: Fine-tuning B3 on adapt, eval on shifted...")
    model_b5 = model_from_config(config_path)
    model_b5.load_state_dict(torch.load(f"{models_dir}/B3_cnn1d_randomized.pt", weights_only=True))
    model_b5, _ = train_model(model_b5, adapt_loader, epochs=ft["epochs"],
                              lr=ft["learning_rate"], device=device)
    torch.save(model_b5.cpu().state_dict(), f"{models_dir}/B5_cnn1d_randomized_finetuned.pt")
    preds, probs, labels = predict_batch(model_b5, shifted_test_loader, device=device)
    metrics = compute_all_metrics(labels, preds, probs)
    print(f"  B5 Macro-F1: {metrics['macro_f1']:.4f}")
    _save_result("B5_cnn1d_randomized_finetune_on_shifted", metrics, results_dir, labels, preds, probs)


def main():
    parser = argparse.ArgumentParser(description="Run B1-B5 classification benchmarks")
    parser.add_argument("--quick", action="store_true", help="Reduced epochs for quick runs")
    parser.add_argument("--seed", type=int, default=42, help="Training seed (default: 42)")
    parser.add_argument("--output-dir", default="results", help="Results directory (default: results)")
    args = parser.parse_args()

    models_dir = "models" if args.output_dir == "results" else os.path.join(args.output_dir, "models")

    run_all_experiments(
        seed=args.seed,
        results_dir=args.output_dir,
        models_dir=models_dir,
        quick=args.quick,
    )


if __name__ == "__main__":
    main()
