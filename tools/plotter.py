"""
Short script to plot training/val loss/accuracy from a log file.
Example usage:
    python3 tools/plotter.py ./log/train_losses.txt "Training Loss" "Steps" "" 10
"""

import argparse
import matplotlib.pyplot as plt


def get_args():
    parser = argparse.ArgumentParser(description="Plot training loss from a file.")
    parser.add_argument(
        "input_file", type=str, help="Path to the input file containing loss values"
    )
    parser.add_argument(
        "--input_file2",
        type=str,
        default=None,
        help="Path to the second input file containing loss values (optional)",
    )
    parser.add_argument(
        "y_axis_label", type=str, help="Label for the y-axis (e.g., 'Loss')"
    )
    parser.add_argument(
        "x_axis_label", type=str, help="Label for the x-axis (e.g., 'Steps')"
    )
    parser.add_argument(
        "output_file",
        type=str,
        nargs="?",
        default="",
        help="Filepath to save the plot to",
    )
    parser.add_argument(
        "plot_every", type=int, nargs="?", default=1, help="Plot every N steps"
    )
    return parser.parse_args()


def plot_values(args, metrics, metrics2=None):
    plt.plot(metrics, marker=None, linestyle="-", label="Curve 1")
    if metrics2 is not None:
        plt.plot(metrics2, marker=None, linestyle="-", label="Curve 2")
        plt.legend()
    plt.xlabel(args.x_axis_label)
    plt.ylabel(args.y_axis_label)
    plt.grid(True)
    plt.show()

    if args.output_file:
        plt.savefig(args.output_file)


def main():
    args = get_args()
    with open(args.input_file, "r") as f:
        metrics = [float(line.strip()) for line in f]
        if args.plot_every > 1:
            metrics = metrics[:: args.plot_every]

    metrics2 = None
    if args.input_file2:
        with open(args.input_file2, "r") as f:
            metrics2 = [float(line.strip()) for line in f]
            if args.plot_every > 1:
                metrics2 = metrics2[:: args.plot_every]

    plot_values(args, metrics, metrics2)


if __name__ == "__main__":
    main()
