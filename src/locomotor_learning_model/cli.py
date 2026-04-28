"""Command-line interface for running the Python simulation."""

from __future__ import annotations

import argparse
from pathlib import Path

from .workflow import run_simulation


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Run the fall risk-aware locomotor learning model and optionally "
            "save reviewer-friendly figures."
        )
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible runs. Omit to use a fresh seed.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=None,
        help="Override the number of learning iterations.",
    )
    parser.add_argument(
        "--split-or-tied",
        choices=("split", "tied"),
        default="split",
        help="Choose the treadmill context used for the simulation.",
    )
    parser.add_argument(
        "--speed-protocol",
        default=None,
        help=(
            "Optional protocol name. Defaults to the manuscript's classic split-belt "
            "protocol for split runs and the four speed changes protocol for tied runs."
        ),
    )
    parser.add_argument(
        "--transition-time",
        type=float,
        default=None,
        help="Optional treadmill transition time override in seconds.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional directory where figures should be saved as PNG files.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip opening interactive figure windows.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the simulation from the command line."""
    args = build_parser().parse_args(argv)
    results = run_simulation(
        seed=args.seed,
        num_iterations=args.iterations,
        split_or_tied=args.split_or_tied,
        speed_protocol=args.speed_protocol,
        transition_time=args.transition_time,
        make_plots=not args.no_plots,
        output_dir=args.output_dir,
    )

    summary = results.summary
    print("Simulation complete.")
    print(f"Split/tied condition: {results.param_fixed['SplitOrTied']}")
    print(f"Speed protocol: {results.param_fixed['speedProtocol']}")
    print(f"Iterations: {results.param_fixed['num_iterations']}")
    print(f"Average energy rate: {summary['average_energy_rate']:.6f}")
    print(f"Final average energy rate: {summary['final_average_energy_rate']:.6f}")
    if args.output_dir is not None:
        print(f"Saved figures to {args.output_dir.resolve()}")
    return 0
