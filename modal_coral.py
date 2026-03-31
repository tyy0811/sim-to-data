"""Run CORAL experiment on Modal GPU.

Usage: modal run modal_coral.py
"""

import os
from pathlib import Path

import modal

app = modal.App("sim-to-data-coral")

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
                "modal_*.py"],
    )
)


@app.function(image=image, gpu="A10G", timeout=1800)
def run_coral(
    source_train_bytes: bytes,
    shifted_train_bytes: bytes,
    shifted_val_bytes: bytes,
    shifted_test_bytes: bytes,
    b3_checkpoint_bytes: bytes,
):
    import json
    import subprocess
    import sys

    os.chdir("/root/sim-to-data")
    os.makedirs("data", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("results/v3", exist_ok=True)

    # Write uploaded data
    for name, data in [
        ("data/source_train.npz", source_train_bytes),
        ("data/shifted_train.npz", shifted_train_bytes),
        ("data/shifted_val.npz", shifted_val_bytes),
        ("data/shifted_test.npz", shifted_test_bytes),
        ("models/B3_cnn1d_randomized.pt", b3_checkpoint_bytes),
    ]:
        with open(name, "wb") as f:
            f.write(data)
        print(f"  Wrote {name} ({len(data)} bytes)")

    env = {**os.environ, "PYTHONPATH": "/root/sim-to-data/src"}

    print("\n=== Running CORAL experiment (A10G) ===")
    subprocess.run(
        [sys.executable, "experiments/run_coral.py", "--device", "cuda"],
        check=True, env=env,
    )

    # Read results
    with open("results/v3/coral_results.json") as f:
        results = json.load(f)

    with open("models/B6_cnn1d_coral.pt", "rb") as f:
        checkpoint = f.read()

    return {"results": results, "checkpoint": checkpoint}


@app.local_entrypoint()
def main():
    import json

    project = Path(__file__).resolve().parent

    # Read local data files
    print("Uploading data to Modal...")
    source_train = (project / "data/source_train.npz").read_bytes()
    shifted_train = (project / "data/shifted_train.npz").read_bytes()
    shifted_val = (project / "data/shifted_val.npz").read_bytes()
    shifted_test = (project / "data/shifted_test.npz").read_bytes()
    b3_ckpt = (project / "models/B3_cnn1d_randomized.pt").read_bytes()

    output = run_coral.remote(
        source_train, shifted_train, shifted_val, shifted_test, b3_ckpt,
    )

    # Save results locally
    os.makedirs("results/v3", exist_ok=True)
    with open("results/v3/coral_results.json", "w") as f:
        json.dump(output["results"], f, indent=2)

    os.makedirs("models", exist_ok=True)
    with open("models/B6_cnn1d_coral.pt", "wb") as f:
        f.write(output["checkpoint"])

    print("\n" + "=" * 60)
    r = output["results"]
    print(f"  B6 CORAL — best weight: {r['best_coral_weight']}")
    print(f"  Val F1:  {r['best_val_f1']:.4f}")
    print(f"  Test F1: {r['test_result']['macro_f1']:.4f}")
    print(f"  Test Acc: {r['test_result']['accuracy']:.4f}")
    print("=" * 60)
    print("\nSaved: results/v3/coral_results.json")
    print("Saved: models/B6_cnn1d_coral.pt")
