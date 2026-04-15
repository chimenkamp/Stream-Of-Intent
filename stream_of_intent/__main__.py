from __future__ import annotations

import argparse
import csv
import sys
from typing import Optional, TextIO

from stream_of_intent.optimization import generate_intentional_stream
from stream_of_intent.types import FeatureVector, StaticParams


def main() -> None:
    """Parse arguments and run the intentional stream generation pipeline."""
    parser = argparse.ArgumentParser(
        prog="stream_of_intent",
        description="Generate synthetic event streams with controllable features.",
    )

    parser.add_argument(
        "--temporal-dependency",
        type=float,
        default=0.5,
        help="Target temporal dependency strength [0, 1]. Default: 0.5",
    )
    parser.add_argument(
        "--long-term-dependency",
        type=float,
        default=0.5,
        help="Target long-term dependency strength [0, 1]. Default: 0.5",
    )
    parser.add_argument(
        "--non-linear-dependency",
        type=float,
        default=0.5,
        help="Target non-linear dependency strength [0, 1]. Default: 0.5",
    )
    parser.add_argument(
        "--out-of-order",
        type=float,
        default=0.0,
        help="Target out-of-order strength [0, 1]. Default: 0.0",
    )
    parser.add_argument(
        "--fractal-behavior",
        type=float,
        default=0.0,
        help="Target fractal behavior strength [0, 1]. Default: 0.0",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=50,
        help="Maximum Bayesian optimization iterations. Default: 50",
    )
    parser.add_argument(
        "--stream-length",
        type=int,
        default=5000,
        help="Number of events to generate. Default: 5000",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=1000,
        help="Events per tumbling window for feature extraction. Default: 1000",
    )
    parser.add_argument(
        "--num-activities",
        type=int,
        default=10,
        help="Number of distinct activities. Default: 10",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility. Default: 42",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV file path. Default: stdout",
    )

    args = parser.parse_args()

    targets = FeatureVector(
        temporal_dependency=args.temporal_dependency,
        long_term_dependency=args.long_term_dependency,
        non_linear_dependency=args.non_linear_dependency,
        out_of_order=args.out_of_order,
        fractal_behavior=args.fractal_behavior,
    )

    static = StaticParams(
        window_size=args.window_size,
        num_activities=args.num_activities,
        stream_length=args.stream_length,
        random_seed=args.seed,
    )

    stream = generate_intentional_stream(
        targets=targets,
        static_params=static,
        max_iterations=args.max_iterations,
    )

    output_file: Optional[TextIO] = None
    try:
        if args.output:
            output_file = open(args.output, "w", newline="")
            writer = csv.writer(output_file)
        else:
            writer = csv.writer(sys.stdout)

        writer.writerow([
            "case_id",
            "activity",
            "timestamp",
            "event_type",
            "arrival_timestamp",
        ])

        for event in stream:
            writer.writerow([
                event.case_id,
                event.activity,
                f"{event.timestamp:.6f}",
                event.event_type,
                f"{event.arrival_timestamp:.6f}",
            ])

    finally:
        if output_file is not None:
            output_file.close()


if __name__ == "__main__":
    main()
