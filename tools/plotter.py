"""
Script to plot training/val loss/accuracy from log file(s).

Example usages:
    python3 tools/plotter.py ./log/train_losses.txt "Training Loss" "Steps" --plot_every 10
    python3 tools/plotter.py ./log/train_losses.txt "Training Loss" "Steps" --plot_every 10 \
        --input_file2 losses.txt --label1 "My Implementation" --label2 "PyTorch"
"""

import argparse
import matplotlib.pyplot as plt
from typing import List, Optional


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot training loss from a file.")
    parser.add_argument(
        "input_file", type=str, help="Path to the input file containing loss values"
    )
    parser.add_argument(
        "y_axis_label", type=str, help="Label for the y-axis (e.g., 'Loss')"
    )
    parser.add_argument(
        "x_axis_label", type=str, help="Label for the x-axis (e.g., 'Steps')"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="",
        help="Filepath to save the plot to (optional)",
    )
    parser.add_argument(
        "--plot_every",
        type=int,
        default=1,
        help="Plot every N steps (default: 1)",
    )
    parser.add_argument(
        "--input_file2",
        type=str,
        default=None,
        help="Path to the second input file containing loss values (optional)",
    )
    parser.add_argument(
        "--label1",
        type=str,
        default="Curve 1",
        help="Legend label for the first curve (default: 'Curve 1')",
    )
    parser.add_argument(
        "--label2",
        type=str,
        default="Curve 2",
        help="Legend label for the second curve if provided (default: 'Curve 2')",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="",
        help="Title for the plot (optional)",
    )
    parser.add_argument(
        "--log_scale",
        action="store_true",
        help="Plot the y-axis on a logarithmic scale",
    )
    return parser.parse_args()


def read_metrics(file_path: str, plot_every: int) -> List[float]:
    with open(file_path, "r") as f:
        metrics = [float(line.strip()) for line in f]
    if plot_every > 1:
        metrics = metrics[::plot_every]
    return metrics


def plot_values(
    args: argparse.Namespace,
    metrics: List[float],
    metrics2: Optional[List[float]] = None,
) -> None:
    plt.plot(metrics, linestyle="-", label=args.label1)
    if metrics2 is not None:
        plt.plot(metrics2, linestyle="-", label=args.label2)
        plt.legend()
    plt.xlabel(args.x_axis_label)
    plt.ylabel(args.y_axis_label)
    if args.log_scale:
        plt.yscale("log")
    if args.title:
        plt.title(args.title)
    plt.grid(True)
    plt.tight_layout()

    if args.output_file:
        plt.savefig(args.output_file)
    plt.show()


def main() -> None:
    args = get_args()
    metrics = read_metrics(args.input_file, args.plot_every)

    metrics2 = None
    if args.input_file2:
        metrics2 = read_metrics(args.input_file2, args.plot_every)

    plot_values(args, metrics, metrics2)


if __name__ == "__main__":
    main()
