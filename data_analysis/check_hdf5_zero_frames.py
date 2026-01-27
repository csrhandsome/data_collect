#!/usr/bin/env python3
"""
Report per-file frame counts and zero-motion (idle) frame counts for HDF5 logs.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check HDF5 frame counts and zero-motion frames."
    )
    parser.add_argument("--input-dir", type=Path, default=Path("data/hdf5"))
    parser.add_argument("--pattern", type=str, default="*.h5")
    parser.add_argument(
        "--epsilon",
        type=float,
        default=1e-6,
        help="Zero-motion threshold for action deltas.",
    )
    parser.add_argument(
        "--epsilons",
        type=float,
        nargs="+",
        default=[1e-8, 1e-6, 1e-4, 1e-3],
        help="Additional thresholds for per-dimension near-zero ratios.",
    )
    parser.add_argument(
        "--hist-bins",
        type=int,
        default=21,
        help="Number of histogram bins per action dimension.",
    )
    parser.add_argument(
        "--hist-range",
        type=float,
        nargs=2,
        metavar=("MIN", "MAX"),
        default=None,
        help="Optional fixed histogram range shared across dimensions.",
    )
    parser.add_argument(
        "--no-hist", action="store_true", help="Skip per-dimension histogram summaries."
    )
    parser.add_argument(
        "--token-bins",
        type=int,
        default=256,
        help="Number of action token bins (OpenVLA default=256).",
    )
    parser.add_argument(
        "--abs-dims",
        type=str,
        default=None,
        help="Comma-separated action dims treated as absolute (skip normalization). "
        "Defaults to last dim when action_dim == 7.",
    )
    parser.add_argument(
        "--per-file-qstats",
        action="store_true",
        help="Use per-file q01/q99 for normalization instead of global stats.",
    )
    parser.add_argument(
        "--no-quant", action="store_true", help="Skip token-bin distribution summaries."
    )
    args = parser.parse_args()

    def format_vec(vec: np.ndarray) -> str:
        return np.array2string(
            vec, precision=6, separator=", ", floatmode="maxprec_equal"
        )

    def compute_stats(actions: np.ndarray) -> dict:
        return {
            "min": actions.min(axis=0),
            "max": actions.max(axis=0),
            "mean": actions.mean(axis=0),
            "std": actions.std(axis=0),
            "q01": np.quantile(actions, 0.01, axis=0),
            "q99": np.quantile(actions, 0.99, axis=0),
        }

    def print_stats(label: str, actions: np.ndarray) -> None:
        stats = compute_stats(actions)
        print(f"{label}: action stats")
        for key in ("min", "max", "mean", "std", "q01", "q99"):
            print(f"  {key}: {format_vec(stats[key])}")
        if actions.shape[1] >= 6:
            abs_actions = np.abs(actions[:, :6])
            print(
                f"  abs_q99[:6]: {format_vec(np.quantile(abs_actions, 0.99, axis=0))}"
            )
            print(f"  abs_max[:6]: {format_vec(abs_actions.max(axis=0))}")

    def parse_abs_dims(action_dim: int) -> np.ndarray:
        if args.abs_dims is None:
            return (
                np.array([action_dim - 1], dtype=int)
                if action_dim == 7
                else np.array([], dtype=int)
            )
        if args.abs_dims.strip() == "":
            return np.array([], dtype=int)
        return np.array([int(x) for x in args.abs_dims.split(",")], dtype=int)

    def compute_zero_ratios(values: np.ndarray) -> str:
        parts = []
        for eps in args.epsilons:
            ratio = float(np.mean(np.abs(values) <= eps))
            parts.append(f"{eps:g}:{ratio:.3f}")
        return " ".join(parts)

    def normalize_for_tokenization(
        actions: np.ndarray,
        stats: dict,
        normalization_mask: np.ndarray,
        bin_min: float,
        bin_max: float,
    ) -> np.ndarray:
        q01 = np.asarray(stats["q01"], dtype=np.float64)
        q99 = np.asarray(stats["q99"], dtype=np.float64)
        denom = (q99 - q01) + 1e-8
        normalized = actions.astype(np.float64, copy=True)
        for dim in range(actions.shape[1]):
            if normalization_mask[dim]:
                normalized[:, dim] = (
                    2 * (normalized[:, dim] - q01[dim]) / denom[dim] - 1
                )
            normalized[:, dim] = np.clip(normalized[:, dim], bin_min, bin_max)
        return normalized

    def token_bin_summary(
        actions: np.ndarray,
        stats: dict,
        normalization_mask: np.ndarray,
        bin_min: float,
        bin_max: float,
    ) -> list[dict]:
        bins = np.linspace(bin_min, bin_max, args.token_bins)
        bin_centers = (bins[:-1] + bins[1:]) / 2.0
        center_idx = len(bin_centers) // 2
        normalized = normalize_for_tokenization(
            actions, stats, normalization_mask, bin_min, bin_max
        )
        summaries = []
        for dim in range(actions.shape[1]):
            discretized = np.digitize(normalized[:, dim], bins)
            discretized = np.clip(discretized - 1, a_min=0, a_max=len(bin_centers) - 1)
            counts = np.bincount(discretized, minlength=len(bin_centers))
            total = counts.sum()
            ratios = counts / total if total else counts.astype(np.float64)
            top_idx = int(np.argmax(ratios)) if total else 0
            summaries.append(
                {
                    "center_ratio": float(ratios[center_idx]) if total else 0.0,
                    "top_ratio": float(ratios[top_idx]) if total else 0.0,
                    "top_center": float(bin_centers[top_idx]) if total else 0.0,
                    "bin_width": float((bin_max - bin_min) / (args.token_bins - 1)),
                }
            )
        return summaries

    def get_hist_range(values: np.ndarray) -> tuple[float, float]:
        if args.hist_range is not None:
            return float(args.hist_range[0]), float(args.hist_range[1])
        min_val = float(values.min())
        max_val = float(values.max())
        if min_val == max_val:
            span = max(1e-6, abs(min_val) * 0.1)
            return min_val - span, max_val + span
        return min_val, max_val

    def format_hist(values: np.ndarray) -> str:
        hist_range = get_hist_range(values)
        counts, edges = np.histogram(values, bins=args.hist_bins, range=hist_range)
        total = counts.sum()
        pct = counts / total if total else counts.astype(np.float64)
        centers = (edges[:-1] + edges[1:]) / 2.0
        return f"centers={format_vec(centers)} pct={format_vec(pct)}"

    def print_histograms(label: str, actions: np.ndarray) -> None:
        if args.no_hist:
            return
        print(f"{label}: per-dimension histograms (bins={args.hist_bins})")
        for dim in range(actions.shape[1]):
            values = actions[:, dim]
            zero_ratio = float(np.mean(np.abs(values) <= args.epsilon))
            print(f"  dim{dim}: abs<=epsilon ratio={zero_ratio:.3f}")
            print(f"    {format_hist(values)}")

    files = sorted(args.input_dir.glob(args.pattern))
    if not files:
        raise FileNotFoundError(
            f"No HDF5 files found in {args.input_dir} matching {args.pattern}"
        )

    total_frames = 0
    total_zero = 0
    all_actions = []
    loaded = []

    for path in files:
        import h5py

        with h5py.File(path, "r") as h5_file:
            if "action" not in h5_file:
                print(f"{path.name}: missing action dataset, skipping")
                continue
            actions = h5_file["action"][:]
        if actions.ndim != 2 or actions.shape[1] < 6:
            print(f"{path.name}: unexpected action shape {actions.shape}, skipping")
            continue

        num_frames = int(actions.shape[0])
        zero_mask = np.all(np.abs(actions[:, :6]) <= args.epsilon, axis=1)
        num_zero = int(np.sum(zero_mask))
        ratio = (num_zero / num_frames) if num_frames else 0.0
        total_frames += num_frames
        total_zero += num_zero
        all_actions.append(actions)
        loaded.append((path, actions, num_frames, num_zero, ratio))

    if not loaded:
        raise FileNotFoundError(
            f"No valid HDF5 files found in {args.input_dir} matching {args.pattern}"
        )

    all_actions_concat = np.concatenate(all_actions, axis=0)
    global_stats = compute_stats(all_actions_concat)
    action_dim = all_actions_concat.shape[1]
    abs_dims = parse_abs_dims(action_dim)
    normalization_mask = np.ones((action_dim,), dtype=bool)
    normalization_mask[abs_dims] = False

    for path, actions, num_frames, num_zero, ratio in loaded:
        print(
            f"{path.name}: frames={num_frames} zero={num_zero} zero_ratio={ratio:.3f}"
        )
        print_stats(path.name, actions)
        print_histograms(path.name, actions)
        print(f"{path.name}: per-dimension near-zero ratios (abs<=epsilon)")
        for dim in range(actions.shape[1]):
            ratios = compute_zero_ratios(actions[:, dim])
            print(f"  dim{dim}: {ratios}")

        if not args.no_quant:
            stats_for_quant = (
                compute_stats(actions) if args.per_file_qstats else global_stats
            )
            summaries = token_bin_summary(
                actions, stats_for_quant, normalization_mask, -1.0, 1.0
            )
            print(
                f"{path.name}: token-bin summary (bins={args.token_bins}, normalized)"
            )
            for dim, summary in enumerate(summaries):
                print(
                    "  dim{dim}: center_ratio={center:.3f} top_ratio={top:.3f} "
                    "top_center={center_val:.6f} bin_width={bin_width:.6f}".format(
                        dim=dim,
                        center=summary["center_ratio"],
                        top=summary["top_ratio"],
                        center_val=summary["top_center"],
                        bin_width=summary["bin_width"],
                    )
                )

    if total_frames:
        total_ratio = total_zero / total_frames
        print(
            f"TOTAL: frames={total_frames} zero={total_zero} zero_ratio={total_ratio:.3f}"
        )
        print_stats("TOTAL", all_actions_concat)
        print_histograms("TOTAL", all_actions_concat)
        print("TOTAL: per-dimension near-zero ratios (abs<=epsilon)")
        for dim in range(all_actions_concat.shape[1]):
            ratios = compute_zero_ratios(all_actions_concat[:, dim])
            print(f"  dim{dim}: {ratios}")
        if not args.no_quant:
            summaries = token_bin_summary(
                all_actions_concat, global_stats, normalization_mask, -1.0, 1.0
            )
            print(f"TOTAL: token-bin summary (bins={args.token_bins}, normalized)")
            for dim, summary in enumerate(summaries):
                print(
                    "  dim{dim}: center_ratio={center:.3f} top_ratio={top:.3f} "
                    "top_center={center_val:.6f} bin_width={bin_width:.6f}".format(
                        dim=dim,
                        center=summary["center_ratio"],
                        top=summary["top_ratio"],
                        center_val=summary["top_center"],
                        bin_width=summary["bin_width"],
                    )
                )


if __name__ == "__main__":
    main()
