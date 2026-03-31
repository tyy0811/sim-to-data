"""CORAL adaptation baseline experiment.

Fine-tunes B3 (randomized pretrained) with CORAL loss on shifted
target data. Sweeps coral_weight, picks best by val F1, reports
one row: B6 = B3 + CORAL fine-tune.

Usage:
    python experiments/run_coral.py
    python experiments/run_coral.py --coral-weights 0.1 0.5 1.0 5.0
"""

import argparse
import copy
import json
import os

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from simtodata.adaptation.coral import train_with_coral
from simtodata.data.dataset import InspectionDataset
from simtodata.data.transforms import Normalize
from simtodata.models.factory import model_from_config
from simtodata.models.predict import predict_batch


def main():
    parser = argparse.ArgumentParser(description="CORAL adaptation baseline")
    parser.add_argument("--checkpoint", default="models/B3_cnn1d_randomized.pt",
                        help="B3 randomized pretrained checkpoint")
    parser.add_argument("--model-config", default="configs/model_cnn1d.yaml")
    parser.add_argument("--source-data", default="data/source_train.npz")
    parser.add_argument("--target-data", default="data/shifted_train.npz")
    parser.add_argument("--val-data", default="data/shifted_val.npz")
    parser.add_argument("--test-data", default="data/shifted_test.npz")
    parser.add_argument("--coral-weights", nargs="+", type=float,
                        default=[0.1, 0.5, 1.0, 5.0])
    parser.add_argument("--n-epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print("CORAL adaptation baseline")
    print("=" * 60)

    # Load data
    norm = Normalize()
    source_ds = InspectionDataset(args.source_data, transform=norm)
    target_ds = InspectionDataset(args.target_data, transform=norm)
    val_ds = InspectionDataset(args.val_data, transform=norm)

    source_loader = DataLoader(source_ds, batch_size=args.batch_size, shuffle=True)
    target_loader = DataLoader(target_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    # Load base model
    base_model = model_from_config(args.model_config)
    base_model.load_state_dict(torch.load(args.checkpoint, weights_only=True))

    best_f1 = -1.0
    best_weight = None
    best_state = None
    sweep_results = []

    for cw in args.coral_weights:
        print(f"\n  coral_weight={cw}")
        model = model_from_config(args.model_config)
        model.load_state_dict(copy.deepcopy(base_model.state_dict()))
        model.to(args.device)

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        history = train_with_coral(
            model, source_loader, target_loader, optimizer,
            feature_layer="features",
            coral_weight=cw,
            n_epochs=args.n_epochs,
            device=args.device,
        )

        # Evaluate on val set — predict_batch returns (preds, probs, labels)
        preds, probs, labels = predict_batch(model, val_loader, device=args.device)
        val_f1 = f1_score(labels, preds, average="macro")
        print(f"    Val F1: {val_f1:.4f}  "
              f"(CE: {history[-1]['ce_loss']:.4f}, "
              f"CORAL: {history[-1]['coral_loss']:.6f})")

        sweep_results.append({
            "coral_weight": cw,
            "val_f1": val_f1,
            "history": history,
        })

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_weight = cw
            best_state = copy.deepcopy(model.state_dict())

    print(f"\n  Best coral_weight: {best_weight} (val F1: {best_f1:.4f})")

    # Evaluate best on test set
    model = model_from_config(args.model_config)
    model.load_state_dict(best_state)
    model.to(args.device)

    test_ds = InspectionDataset(args.test_data, transform=norm)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)
    test_preds, test_probs, test_labels = predict_batch(
        model, test_loader, device=args.device,
    )
    test_f1 = f1_score(test_labels, test_preds, average="macro")
    test_acc = (test_preds == test_labels).mean()

    print(f"\n  B6 test result:")
    print(f"    Macro F1: {test_f1:.4f}")
    print(f"    Accuracy: {test_acc:.4f}")

    # Save
    v3_dir = os.path.join(args.results_dir, "v3")
    os.makedirs(v3_dir, exist_ok=True)
    test_result = {"macro_f1": float(test_f1), "accuracy": float(test_acc)}
    out = {
        "name": "B6_coral",
        "best_coral_weight": best_weight,
        "best_val_f1": float(best_f1),
        "test_result": test_result,
        "sweep": [{k: v for k, v in r.items() if k != "history"}
                  for r in sweep_results],
    }
    out_path = os.path.join(v3_dir, "coral_results.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\n  Saved: {out_path}")

    # Save best checkpoint
    ckpt_path = os.path.join("models", "B6_cnn1d_coral.pt")
    torch.save(best_state, ckpt_path)
    print(f"  Checkpoint: {ckpt_path}")


if __name__ == "__main__":
    main()
