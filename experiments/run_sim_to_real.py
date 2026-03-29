"""Sim-to-real B-scan experiments: SB1-SB3, SR1-SR2.

SB1: Synthetic source -> synthetic source test
SB2: Synthetic source -> synthetic shifted test
SB3: Synthetic randomized -> synthetic shifted test
SR1: Synthetic source -> real (Virkkunen)
SR2: Synthetic randomized -> real (Virkkunen)
"""

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from simtodata.data.bscan_dataset import BscanDataset, resize_bscan
from simtodata.data.constants import BSCAN_SHAPE
from simtodata.models.cnn2d_bscan import BscanCNN
from simtodata.simulator.bscan import generate_bscan_dataset
from simtodata.simulator.regime import load_regimes_from_yaml


def _seed_everything(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)


def _resize_batch(bscans: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Resize a batch of B-scans."""
    return np.array([resize_bscan(b, shape) for b in bscans])


def _train_bscan_cnn(train_dataset, val_dataset, epochs=50, lr=1e-3, batch_size=64,
                      patience=10, device="cpu"):
    """Train a BscanCNN on a binary B-scan dataset."""
    model = BscanCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    best_val_loss = float("inf")
    best_state = None
    no_improve = 0

    for epoch in range(epochs):
        model.train()
        total_loss, n = 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(y)
            n += len(y)

        # Validation
        model.eval()
        val_loss, correct, val_n = 0, 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                val_loss += criterion(logits, y).item() * len(y)
                correct += (logits.argmax(1) == y).sum().item()
                val_n += len(y)

        val_loss /= val_n
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            print(f"  Early stopping at epoch {epoch + 1}")
            break

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{epochs} - loss: {total_loss/n:.4f} "
                  f"- val_loss: {val_loss:.4f} - val_acc: {correct/val_n:.3f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def _evaluate(model, dataset, batch_size=64, device="cpu"):
    """Evaluate a trained model. Returns dict with accuracy, f1, auroc."""
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.to(device)
    model.eval()
    all_preds, all_probs, all_labels = [], [], []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            probs = torch.softmax(logits, dim=1)
            all_preds.append(logits.argmax(1).cpu().numpy())
            all_probs.append(probs[:, 1].cpu().numpy())  # P(flaw)
            all_labels.append(y.numpy())

    preds = np.concatenate(all_preds)
    probs = np.concatenate(all_probs)
    labels = np.concatenate(all_labels)

    metrics = {
        "accuracy": float(accuracy_score(labels, preds)),
        "f1": float(f1_score(labels, preds, zero_division=0)),
        "n_samples": int(len(labels)),
        "n_flaw": int((labels == 1).sum()),
        "n_noflaw": int((labels == 0).sum()),
    }
    try:
        metrics["auroc"] = float(roc_auc_score(labels, probs))
    except ValueError:
        metrics["auroc"] = float("nan")
    return metrics


def _save_result(name, metrics, results_dir):
    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, f"{name}.json")
    with open(path, "w") as f:
        json.dump({"name": name, "metrics": metrics}, f, indent=2)
    print(f"  -> {path}")


def main():
    parser = argparse.ArgumentParser(description="Sim-to-real B-scan experiments")
    parser.add_argument("--config", default="configs/simulator.yaml")
    parser.add_argument("--real-data-dir", default=None,
                        help="Path to cloned iikka-v/ML-NDT data directory")
    parser.add_argument("--output-dir", default="results/sim_to_real")
    parser.add_argument("--models-dir", default="models/sim_to_real")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quick", action="store_true",
                        help="Small datasets and few epochs for testing")
    args = parser.parse_args()

    _seed_everything(args.seed)
    regimes = load_regimes_from_yaml(args.config)
    os.makedirs(args.models_dir, exist_ok=True)

    n_train = 500 if args.quick else 5000
    n_test = 100 if args.quick else 1000
    n_positions = 32 if args.quick else 64
    epochs = 5 if args.quick else 50

    # --- Generate synthetic B-scan datasets ---
    print(f"Generating synthetic source B-scans ({n_train} train, {n_test} test)...")
    t0 = time.time()
    source_data = generate_bscan_dataset(
        regimes["source"], n_train, seed=args.seed, n_positions=n_positions,
    )
    source_test_data = generate_bscan_dataset(
        regimes["source"], n_test, seed=args.seed + 1, n_positions=n_positions,
    )
    print(f"  Generated in {time.time() - t0:.1f}s")

    print(f"Generating synthetic shifted B-scans ({n_test} test)...")
    shifted_test_data = generate_bscan_dataset(
        regimes["shifted"], n_test, seed=args.seed + 2, n_positions=n_positions,
    )

    print(f"Generating synthetic randomized B-scans ({n_train} train)...")
    rand_data = generate_bscan_dataset(
        regimes["randomized"], n_train, seed=args.seed + 3, n_positions=n_positions,
    )

    # --- Resize all to BSCAN_SHAPE ---
    print(f"Resizing to {BSCAN_SHAPE}...")
    source_bscans = _resize_batch(source_data["bscans"], BSCAN_SHAPE)
    source_test_bscans = _resize_batch(source_test_data["bscans"], BSCAN_SHAPE)
    shifted_test_bscans = _resize_batch(shifted_test_data["bscans"], BSCAN_SHAPE)
    rand_bscans = _resize_batch(rand_data["bscans"], BSCAN_SHAPE)

    # --- Create datasets ---
    # Split source train into train/val (90/10)
    source_full = BscanDataset(source_bscans, source_data["labels"])
    val_size = max(1, len(source_full) // 10)
    train_size = len(source_full) - val_size
    source_train_ds, source_val_ds = random_split(
        source_full, [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed),
    )

    source_test_ds = BscanDataset(source_test_bscans, source_test_data["labels"])
    shifted_test_ds = BscanDataset(shifted_test_bscans, shifted_test_data["labels"])

    rand_full = BscanDataset(rand_bscans, rand_data["labels"])
    rand_train_ds, rand_val_ds = random_split(
        rand_full, [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed),
    )

    # --- SB1: Source -> Source ---
    print("\nSB1: Train on synthetic source, eval on synthetic source test...")
    model_source = _train_bscan_cnn(source_train_ds, source_val_ds, epochs=epochs)
    torch.save(model_source.cpu().state_dict(),
               os.path.join(args.models_dir, "bscan_source.pt"))
    sb1 = _evaluate(model_source, source_test_ds)
    print(f"  SB1 F1={sb1['f1']:.3f}  AUROC={sb1['auroc']:.3f}")
    _save_result("SB1_bscan_source_on_source", sb1, args.output_dir)

    # --- SB2: Source -> Shifted ---
    print("SB2: Eval source model on synthetic shifted test...")
    sb2 = _evaluate(model_source, shifted_test_ds)
    print(f"  SB2 F1={sb2['f1']:.3f}  AUROC={sb2['auroc']:.3f}")
    _save_result("SB2_bscan_source_on_shifted", sb2, args.output_dir)

    # --- SB3: Randomized -> Shifted ---
    print("SB3: Train on synthetic randomized, eval on synthetic shifted test...")
    model_rand = _train_bscan_cnn(rand_train_ds, rand_val_ds, epochs=epochs)
    torch.save(model_rand.cpu().state_dict(),
               os.path.join(args.models_dir, "bscan_randomized.pt"))
    sb3 = _evaluate(model_rand, shifted_test_ds)
    print(f"  SB3 F1={sb3['f1']:.3f}  AUROC={sb3['auroc']:.3f}")
    _save_result("SB3_bscan_randomized_on_shifted", sb3, args.output_dir)

    # --- SR1/SR2: Sim-to-Real (if real data available) ---
    if args.real_data_dir and os.path.isdir(args.real_data_dir):
        from simtodata.data.virkkunen import VirkkunenLoader

        print(f"\nLoading real data from {args.real_data_dir}...")
        loader = VirkkunenLoader(args.real_data_dir)
        real_bscans, real_labels = loader.load_all()
        print(f"  Loaded {len(real_labels)} samples "
              f"(flaw: {(real_labels==1).sum()}, no-flaw: {(real_labels==0).sum()})")

        real_bscans_resized = _resize_batch(real_bscans, BSCAN_SHAPE)
        real_ds = BscanDataset(real_bscans_resized, real_labels)

        print("SR1: Eval source model on real data...")
        sr1 = _evaluate(model_source, real_ds)
        print(f"  SR1 F1={sr1['f1']:.3f}  AUROC={sr1['auroc']:.3f}")
        _save_result("SR1_bscan_source_on_real", sr1, args.output_dir)

        print("SR2: Eval randomized model on real data...")
        sr2 = _evaluate(model_rand, real_ds)
        print(f"  SR2 F1={sr2['f1']:.3f}  AUROC={sr2['auroc']:.3f}")
        _save_result("SR2_bscan_randomized_on_real", sr2, args.output_dir)
    else:
        print("\nNo real data directory provided (--real-data-dir). "
              "Skipping SR1/SR2 sim-to-real experiments.")

    # --- Summary ---
    print("\n=== Results Summary ===")
    for name, m in [("SB1", sb1), ("SB2", sb2), ("SB3", sb3)]:
        print(f"  {name}: F1={m['f1']:.3f}  AUROC={m['auroc']:.3f}")
    if args.real_data_dir and os.path.isdir(args.real_data_dir):
        for name, m in [("SR1", sr1), ("SR2", sr2)]:
            print(f"  {name}: F1={m['f1']:.3f}  AUROC={m['auroc']:.3f}")


if __name__ == "__main__":
    main()
