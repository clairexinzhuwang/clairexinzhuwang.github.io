from __future__ import annotations

import argparse
from pathlib import Path

from ranking_lab.experiment import run_experiment


def build_parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="Run the MovieLens BPR training-budget benchmark."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=root / "configs" / "experiment.json",
        help="Path to the experiment JSON configuration.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: <project>/results).",
    )
    parser.add_argument(
        "--allow-insecure-download",
        action="store_true",
        help=(
            "Disable TLS certificate verification for the GroupLens download. "
            "Use only after checking the official URL; the published archive MD5 "
            "is still enforced."
        ),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = run_experiment(
        config_path=args.config,
        output_dir=args.output_dir,
        allow_insecure_download=args.allow_insecure_download,
    )
    print(result["title"])
    for setting in result["settings"]:
        print(
            f"{setting['setting_id']}: "
            f"accuracy={setting['heldout_pairwise_accuracy']['mean']:.4f}, "
            f"NDCG@10={setting['ndcg_at_10']['mean']:.4f}, "
            f"runtime={setting['runtime_seconds']['mean']:.2f}s, "
            f"top10_jaccard={setting['ranking_stability']['top10_jaccard_mean']:.4f}"
        )


if __name__ == "__main__":
    main()
