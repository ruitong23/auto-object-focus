"""Command line interface for the auto focus controller."""
from __future__ import annotations

import argparse
import sys
from typing import Optional

from .controller import AutoFocusController


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """Parse command line arguments for the CLI."""

    parser = argparse.ArgumentParser(description="Automatically focus the cursor on a detected object.")
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument("--class", dest="target_class", help="Name of the class to track (e.g. 'person').")
    group.add_argument("--class-id", dest="target_class_id", type=int, help="Numeric class id to track.")
    parser.add_argument("--confidence", type=float, default=0.5, help="Minimum detection confidence.")
    parser.add_argument(
        "--smoothing",
        type=float,
        default=0.2,
        help="Smoothing factor between 0 and 1 controlling cursor responsiveness.",
    )
    parser.add_argument(
        "--tracking-speed",
        type=float,
        default=0.2,
        help="Base relative tracking speed (fraction of the screen per update) applied even when the target is near the center.",
    )
    parser.add_argument(
        "--distance-ratio",
        type=float,
        default=2.0,
        help="Additional speed scaling based on how far the detection is from the center.",
    )
    parser.add_argument(
        "--model",
        default="yolov8n.pt",
        help="Path to the YOLO model weights (default: yolov8n.pt).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    """CLI entry point."""

    args = parse_args(argv)
    target = args.target_class if args.target_class is not None else args.target_class_id

    controller = AutoFocusController(
        target_class=target,
        confidence_threshold=args.confidence,
        smoothing_factor=args.smoothing,
        tracking_speed=args.tracking_speed,
        distance_ratio=args.distance_ratio,
        model_path=args.model,
    )

    try:
        controller.run()
    except KeyboardInterrupt:
        print("Interrupted by user. Shutting down...", file=sys.stderr)
    finally:
        controller.close()

    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
