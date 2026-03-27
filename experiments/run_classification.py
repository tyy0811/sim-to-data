"""Run classification benchmark experiments B1-B5."""

import json
import os
import sys

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


def _save_result(name, metrics, results_dir, y_true=None, y_pred=None):
    os.makedirs(results_dir, exist_ok=True)
    result = {"name": name, "metrics": metrics}
    if y_true is not None:
        result["y_true"] = y_true.tolist()
    if y_pred is not None:
        result["y_pred"] = y_pred.tolist()
    path = os.path.join(results_dir, f"{name}.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Saved: {path}")


SEED = 42


def _seed_everything(seed: int):
    """Set all random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _seeded_loader(dataset, batch_size, shuffle):
    """Create a DataLoader with a seeded generator for reproducible shuffling."""
    generator = torch.Generator()
    generator.manual_seed(SEED)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=generator)


def main():
    _seed_everything(SEED)

    config_path = "configs/model_cnn1d.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)
    tc = config["training"]
    ft = config["finetune"]
    results_dir = "results"
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)

    norm = Normalize()
    source_train = InspectionDataset("data/source_train.npz", transform=norm)
    source_val = InspectionDataset("data/source_val.npz", transform=norm)
    source_test = InspectionDataset("data/source_test.npz", transform=norm)
    shifted_test = InspectionDataset("data/shifted_test.npz", transform=norm)
    adapt_data = InspectionDataset("data/shifted_adapt.npz", transform=norm)
    randomized_train = InspectionDataset("data/randomized_train.npz", transform=norm)

    bs = tc["batch_size"]
    train_loader = _seeded_loader(source_train, bs, shuffle=True)
    val_loader = _seeded_loader(source_val, bs, shuffle=False)
    source_test_loader = _seeded_loader(source_test, bs, shuffle=False)
    shifted_test_loader = _seeded_loader(shifted_test, bs, shuffle=False)
    adapt_loader = _seeded_loader(adapt_data, bs, shuffle=True)
    rand_train_loader = _seeded_loader(randomized_train, bs, shuffle=True)

    # B1: Source -> Source
    print("B1: Training on source, eval on source...")
    model_b1 = model_from_config(config_path)
    model_b1, _ = train_model(model_b1, train_loader, val_loader, epochs=tc["epochs"],
                              lr=tc["learning_rate"], weight_decay=tc["weight_decay"],
                              patience=tc["early_stopping_patience"],
                              scheduler_patience=tc["scheduler_patience"],
                              scheduler_factor=tc["scheduler_factor"])
    torch.save(model_b1.state_dict(), f"{models_dir}/B1_cnn1d_source.pt")
    preds, probs, labels = predict_batch(model_b1, source_test_loader)
    metrics = compute_all_metrics(labels, preds, probs)
    print(f"  B1 Macro-F1: {metrics['macro_f1']:.4f}")
    _save_result("B1_cnn1d_source_on_source", metrics, results_dir, labels, preds)

    # B2: Source -> Shifted (reuse B1 model)
    print("B2: Eval B1 on shifted...")
    preds, probs, labels = predict_batch(model_b1, shifted_test_loader)
    metrics = compute_all_metrics(labels, preds, probs)
    print(f"  B2 Macro-F1: {metrics['macro_f1']:.4f}")
    _save_result("B2_cnn1d_source_on_shifted", metrics, results_dir, labels, preds)

    # B3: Randomized -> Shifted
    print("B3: Training on randomized, eval on shifted...")
    model_b3 = model_from_config(config_path)
    model_b3, _ = train_model(model_b3, rand_train_loader, val_loader, epochs=tc["epochs"],
                              lr=tc["learning_rate"], weight_decay=tc["weight_decay"],
                              patience=tc["early_stopping_patience"],
                              scheduler_patience=tc["scheduler_patience"],
                              scheduler_factor=tc["scheduler_factor"])
    torch.save(model_b3.state_dict(), f"{models_dir}/B3_cnn1d_randomized.pt")
    preds, probs, labels = predict_batch(model_b3, shifted_test_loader)
    metrics = compute_all_metrics(labels, preds, probs)
    print(f"  B3 Macro-F1: {metrics['macro_f1']:.4f}")
    _save_result("B3_cnn1d_randomized_on_shifted", metrics, results_dir, labels, preds)

    # B4: Source + fine-tune -> Shifted
    print("B4: Fine-tuning B1 on adapt, eval on shifted...")
    model_b4 = model_from_config(config_path)
    model_b4.load_state_dict(torch.load(f"{models_dir}/B1_cnn1d_source.pt", weights_only=True))
    model_b4, _ = train_model(model_b4, adapt_loader, epochs=ft["epochs"],
                              lr=ft["learning_rate"])
    torch.save(model_b4.state_dict(), f"{models_dir}/B4_cnn1d_source_finetuned.pt")
    preds, probs, labels = predict_batch(model_b4, shifted_test_loader)
    metrics = compute_all_metrics(labels, preds, probs)
    print(f"  B4 Macro-F1: {metrics['macro_f1']:.4f}")
    _save_result("B4_cnn1d_source_finetune_on_shifted", metrics, results_dir, labels, preds)

    # B5: Randomized + fine-tune -> Shifted
    print("B5: Fine-tuning B3 on adapt, eval on shifted...")
    model_b5 = model_from_config(config_path)
    model_b5.load_state_dict(torch.load(f"{models_dir}/B3_cnn1d_randomized.pt", weights_only=True))
    model_b5, _ = train_model(model_b5, adapt_loader, epochs=ft["epochs"],
                              lr=ft["learning_rate"])
    torch.save(model_b5.state_dict(), f"{models_dir}/B5_cnn1d_randomized_finetuned.pt")
    preds, probs, labels = predict_batch(model_b5, shifted_test_loader)
    metrics = compute_all_metrics(labels, preds, probs)
    print(f"  B5 Macro-F1: {metrics['macro_f1']:.4f}")
    _save_result("B5_cnn1d_randomized_finetune_on_shifted", metrics, results_dir, labels, preds)


if __name__ == "__main__":
    main()
