"""Run full benchmark pipeline on Modal GPU.

Usage: modal run modal_train.py
"""

import os
from pathlib import Path

import modal

app = modal.App("sim-to-data-benchmark")

LOCAL_PROJECT_DIR = str(Path(__file__).resolve().parent)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "numpy>=1.24",
        "scipy>=1.10",
        "torch>=2.0",
        "pyyaml>=6.0",
        "scikit-learn>=1.2",
        "matplotlib>=3.7",
        "joblib>=1.2",
    )
    .env({"PYTHONPATH": "/root/sim-to-data/src"})
    .add_local_dir(
        LOCAL_PROJECT_DIR,
        remote_path="/root/sim-to-data",
        ignore=["__pycache__", ".git", ".eggs", "data/", "models/", "results/",
                "docs/figures/", ".ruff_cache", ".pytest_cache", "*.egg-info",
                "modal_train.py"],
    )
)


@app.function(image=image, gpu="T4", timeout=3600)
def run_benchmark():
    import json
    import subprocess
    import sys

    import numpy as np
    import torch
    import yaml
    from torch.utils.data import DataLoader

    from simtodata.data.dataset import InspectionDataset
    from simtodata.data.transforms import Normalize
    from simtodata.evaluation.metrics import compute_all_metrics
    from simtodata.models.factory import model_from_config
    from simtodata.models.predict import predict_batch
    from simtodata.models.train import train_model

    os.chdir("/root/sim-to-data")
    env = {**os.environ, "PYTHONPATH": "/root/sim-to-data/src"}

    # Generate datasets
    print("=== Generating datasets ===")
    subprocess.run([sys.executable, "-m", "simtodata.data.generate"], check=True, env=env)

    # Run baselines
    print("\n=== Running baselines B0a-B0c ===")
    subprocess.run([sys.executable, "experiments/run_baselines.py"], check=True, env=env)

    # CNN B1-B5 on GPU
    print("\n=== Running CNN B1-B5 (GPU) ===")
    SEED = 42
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    config_path = "configs/model_cnn1d.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)
    tc = config["training"]
    ft = config["finetune"]
    os.makedirs("results", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    norm = Normalize()
    source_train = InspectionDataset("data/source_train.npz", transform=norm)
    source_val = InspectionDataset("data/source_val.npz", transform=norm)
    source_test = InspectionDataset("data/source_test.npz", transform=norm)
    shifted_test = InspectionDataset("data/shifted_test.npz", transform=norm)
    adapt_data = InspectionDataset("data/shifted_adapt.npz", transform=norm)
    randomized_train = InspectionDataset("data/randomized_train.npz", transform=norm)

    bs = tc["batch_size"]

    def loader(ds, shuffle):
        g = torch.Generator()
        g.manual_seed(SEED)
        return DataLoader(ds, batch_size=bs, shuffle=shuffle, generator=g)

    train_loader = loader(source_train, True)
    val_loader = loader(source_val, False)
    source_test_loader = loader(source_test, False)
    shifted_test_loader = loader(shifted_test, False)
    adapt_loader = loader(adapt_data, True)
    rand_train_loader = loader(randomized_train, True)

    def save_result(name, metrics, y_true=None, y_pred=None, y_proba=None):
        result = {"name": name, "metrics": metrics}
        if y_true is not None:
            result["y_true"] = y_true.tolist()
        if y_pred is not None:
            result["y_pred"] = y_pred.tolist()
        if y_proba is not None:
            result["y_proba"] = y_proba.tolist()
        with open(f"results/{name}.json", "w") as f:
            json.dump(result, f, indent=2)

    # B1
    print("B1: Training on source...")
    model_b1 = model_from_config(config_path)
    model_b1, hist = train_model(model_b1, train_loader, val_loader, epochs=tc["epochs"],
                                  lr=tc["learning_rate"], weight_decay=tc["weight_decay"],
                                  patience=tc["early_stopping_patience"],
                                  scheduler_patience=tc["scheduler_patience"],
                                  scheduler_factor=tc["scheduler_factor"], device=device)
    print(f"  B1 trained {len(hist['train_loss'])} epochs")
    torch.save(model_b1.cpu().state_dict(), "models/B1_cnn1d_source.pt")
    preds, probs, labels = predict_batch(model_b1, source_test_loader, device=device)
    metrics = compute_all_metrics(labels, preds, probs)
    print(f"  B1 F1: {metrics['macro_f1']:.4f}")
    save_result("B1_cnn1d_source_on_source", metrics, labels, preds, probs)

    # B2
    print("B2: Eval B1 on shifted...")
    preds, probs, labels = predict_batch(model_b1, shifted_test_loader, device=device)
    metrics = compute_all_metrics(labels, preds, probs)
    print(f"  B2 F1: {metrics['macro_f1']:.4f}")
    save_result("B2_cnn1d_source_on_shifted", metrics, labels, preds, probs)

    # B3
    print("B3: Training on randomized...")
    model_b3 = model_from_config(config_path)
    model_b3, hist = train_model(model_b3, rand_train_loader, val_loader, epochs=tc["epochs"],
                                  lr=tc["learning_rate"], weight_decay=tc["weight_decay"],
                                  patience=tc["early_stopping_patience"],
                                  scheduler_patience=tc["scheduler_patience"],
                                  scheduler_factor=tc["scheduler_factor"], device=device)
    print(f"  B3 trained {len(hist['train_loss'])} epochs")
    torch.save(model_b3.cpu().state_dict(), "models/B3_cnn1d_randomized.pt")
    preds, probs, labels = predict_batch(model_b3, shifted_test_loader, device=device)
    metrics = compute_all_metrics(labels, preds, probs)
    print(f"  B3 F1: {metrics['macro_f1']:.4f}")
    save_result("B3_cnn1d_randomized_on_shifted", metrics, labels, preds, probs)

    # B4
    print("B4: Fine-tuning B1...")
    model_b4 = model_from_config(config_path)
    model_b4.load_state_dict(torch.load("models/B1_cnn1d_source.pt", weights_only=True))
    model_b4, _ = train_model(model_b4, adapt_loader, epochs=ft["epochs"],
                               lr=ft["learning_rate"], device=device)
    torch.save(model_b4.cpu().state_dict(), "models/B4_cnn1d_source_finetuned.pt")
    preds, probs, labels = predict_batch(model_b4, shifted_test_loader, device=device)
    metrics = compute_all_metrics(labels, preds, probs)
    print(f"  B4 F1: {metrics['macro_f1']:.4f}")
    save_result("B4_cnn1d_source_finetune_on_shifted", metrics, labels, preds, probs)

    # B5
    print("B5: Fine-tuning B3...")
    model_b5 = model_from_config(config_path)
    model_b5.load_state_dict(torch.load("models/B3_cnn1d_randomized.pt", weights_only=True))
    model_b5, _ = train_model(model_b5, adapt_loader, epochs=ft["epochs"],
                               lr=ft["learning_rate"], device=device)
    torch.save(model_b5.cpu().state_dict(), "models/B5_cnn1d_randomized_finetuned.pt")
    preds, probs, labels = predict_batch(model_b5, shifted_test_loader, device=device)
    metrics = compute_all_metrics(labels, preds, probs)
    print(f"  B5 F1: {metrics['macro_f1']:.4f}")
    save_result("B5_cnn1d_randomized_finetune_on_shifted", metrics, labels, preds, probs)

    # Robustness sweep
    print("\n=== Running robustness sweep ===")
    subprocess.run([sys.executable, "experiments/run_robustness.py"], check=True, env=env)

    # Adaptation curve
    print("\n=== Running adaptation curve ===")
    subprocess.run([sys.executable, "experiments/run_adaptation_curve.py"], check=True, env=env)

    # Generate figures
    print("\n=== Generating figures ===")
    subprocess.run([sys.executable, "experiments/generate_figures.py"], check=True, env=env)

    # Collect all results
    all_results = {}
    for name in ["B0a_logreg_source_on_source", "B0b_gb_source_on_source",
                  "B0c_gb_source_on_shifted", "B1_cnn1d_source_on_source",
                  "B2_cnn1d_source_on_shifted", "B3_cnn1d_randomized_on_shifted",
                  "B4_cnn1d_source_finetune_on_shifted",
                  "B5_cnn1d_randomized_finetune_on_shifted"]:
        with open(f"results/{name}.json") as f:
            all_results[name] = json.load(f)

    for extra in ["robustness_sweep", "adaptation_curve"]:
        with open(f"results/{extra}.json") as f:
            all_results[extra] = json.load(f)

    return all_results


@app.local_entrypoint()
def main():
    import json

    results = run_benchmark.remote()
    os.makedirs("results", exist_ok=True)

    print("\n" + "=" * 60)
    print("  BENCHMARK RESULTS")
    print("=" * 60)
    benchmark_names = [
        "B0a_logreg_source_on_source", "B0b_gb_source_on_source",
        "B0c_gb_source_on_shifted", "B1_cnn1d_source_on_source",
        "B2_cnn1d_source_on_shifted", "B3_cnn1d_randomized_on_shifted",
        "B4_cnn1d_source_finetune_on_shifted",
        "B5_cnn1d_randomized_finetune_on_shifted",
    ]
    print(f"\n{'ID':<45} {'F1':>6} {'AUROC':>7} {'ECE':>7}")
    print("-" * 68)
    for name in benchmark_names:
        m = results[name]["metrics"]
        print(f"{name:<45} {m['macro_f1']:>6.4f} {m['auroc']:>7.4f} {m['ece']:>7.4f}")
        with open(f"results/{name}.json", "w") as f:
            json.dump(results[name], f, indent=2)

    for extra in ["robustness_sweep", "adaptation_curve"]:
        with open(f"results/{extra}.json", "w") as f:
            json.dump(results[extra], f, indent=2)

    print("\nAll results saved locally to results/")
