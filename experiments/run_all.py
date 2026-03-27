"""Single entry point for the full benchmark pipeline."""

import argparse
import subprocess
import sys
import time


def run(cmd, description):
    print(f"\n{'='*60}")
    print(f"  {description}")
    print(f"{'='*60}\n")
    t0 = time.time()
    result = subprocess.run([sys.executable] + cmd.split(), cwd=".")
    if result.returncode != 0:
        print(f"FAILED: {description}")
        sys.exit(1)
    print(f"\n  Completed in {time.time()-t0:.1f}s")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Small datasets, 2 epochs")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    gen_args = "-m simtodata.data.generate"
    if args.quick:
        gen_args += " --quick"

    run(gen_args, "Generating datasets")
    run("experiments/run_baselines.py", "Running baselines B0a-B0c")
    run("experiments/run_classification.py", "Running CNN experiments B1-B5")
    run("experiments/run_robustness.py", "Running robustness sweep")
    run("experiments/run_adaptation_curve.py", "Running adaptation curve")
    run("experiments/generate_figures.py", "Generating figures")

    print("\n" + "=" * 60)
    print("  PIPELINE COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
