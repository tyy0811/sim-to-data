"""Run B1-B5 across 5 training seeds on a fixed dataset."""

import argparse
import os

from run_classification import run_all_experiments

SEEDS = [42, 123, 456, 789, 1024]


def main():
    parser = argparse.ArgumentParser(description="Multi-seed B1-B5 benchmark")
    parser.add_argument("--quick", action="store_true", help="Reduced epochs")
    parser.add_argument("--output-dir", default="results/multiseed",
                        help="Base output directory (default: results/multiseed)")
    args = parser.parse_args()

    for seed in SEEDS:
        print(f"\n{'='*50}")
        print(f"  Seed {seed}")
        print(f"{'='*50}")
        seed_dir = os.path.join(args.output_dir, f"seed_{seed}")
        run_all_experiments(
            seed=seed,
            results_dir=seed_dir,
            models_dir=os.path.join(seed_dir, "models"),
            quick=args.quick,
        )

    print(f"\nAll seeds complete. Results in {args.output_dir}/")


if __name__ == "__main__":
    main()
