"""
main.py — Thin CLI entry point for all LoP experiments.

Usage:
    python main.py cifar   -c lop/incremental_cifar/cfg/base.json --index 0
    python main.py mnist   -c lop/permuted_mnist/cfg/bp.json
    python main.py rl      -c lop/rl/cfg/ant/std.yml --seed 1
    python main.py regression -c lop/slowly_changing_regression/cfg/base.json
    python main.py imagenet   -c lop/imagenet/cfg/base.json
"""

import sys
import argparse


def main():
    parser = argparse.ArgumentParser(
        description="LoP Experiment Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "experiment",
        choices=["cifar", "mnist", "rl", "regression", "imagenet"],
        help="Which experiment to run.",
    )
    parser.add_argument(
        "-c", "--config",
        type=str,
        required=True,
        help="Path to the JSON/YAML config file for the experiment.",
    )

    # Pass remaining args to sub-experiments
    args, remaining = parser.parse_known_args()

    if args.experiment == "cifar":
        # Incremental CIFAR-100
        sys.argv = [sys.argv[0], "-c", args.config] + remaining
        from lop.incremental_cifar.incremental_cifar_experiment import main as run
        run(sys.argv[1:])

    elif args.experiment == "mnist":
        # Online Permuted MNIST
        sys.argv = [sys.argv[0], "-c", args.config] + remaining
        from lop.permuted_mnist.online_expr import main as run
        run(sys.argv[1:])

    elif args.experiment == "rl":
        # PPO-based RL
        sys.argv = [sys.argv[0], "-c", args.config] + remaining
        from lop.rl.run_ppo import main as run
        run(sys.argv[1:])

    elif args.experiment == "regression":
        # Slowly Changing Regression
        sys.argv = [sys.argv[0], "-c", args.config] + remaining
        from lop.slowly_changing_regression.expr import main as run
        run(sys.argv[1:])

    elif args.experiment == "imagenet":
        # ImageNet
        sys.argv = [sys.argv[0], "-c", args.config] + remaining
        from lop.imagenet.single_expr import main as run
        run(sys.argv[1:])


if __name__ == "__main__":
    main()
