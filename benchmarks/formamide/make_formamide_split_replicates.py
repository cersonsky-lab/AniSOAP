from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import numpy as np
from ase.io import read, write


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create stratified formamide train/validation/test split replicates."
    )
    parser.add_argument("--input-cg", type=Path, default=Path("formamide_cg.xyz"))
    parser.add_argument("--current-split-dir", type=Path, default=Path("publication_splits"))
    parser.add_argument("--output-root", type=Path, default=Path("split_replicates"))
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--n-bins", type=int, default=10)
    parser.add_argument("--seed-base", type=int, default=20260800)
    parser.add_argument("--train-frac", type=float, default=0.70)
    parser.add_argument("--valid-frac", type=float, default=0.15)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def interaction_energy(frame) -> float:
    return float(frame.info["interaction_energy"])


def rank_bins(values: np.ndarray, n_bins: int) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    bins = np.empty(len(values), dtype=int)
    chunks = np.array_split(order, n_bins)
    for b, idx in enumerate(chunks):
        bins[idx] = b
    return bins


def allocate_counts(
    bin_sizes: list[int],
    total_count: int,
    fraction: float,
) -> list[int]:
    raw = np.asarray(bin_sizes, dtype=float) * fraction
    counts = np.floor(raw).astype(int)
    needed = int(total_count - counts.sum())

    if needed > 0:
        order = np.argsort(-(raw - counts))
        for i in order[:needed]:
            counts[i] += 1
    elif needed < 0:
        order = np.argsort(raw - counts)
        for i in order[: -needed]:
            if counts[i] <= 0:
                continue
            counts[i] -= 1

    if counts.sum() != total_count:
        raise RuntimeError(
            f"Could not allocate exact count {total_count}; got {counts.sum()}"
        )

    return counts.tolist()

def write_replicate_split(
    *,
    split_id: str,
    seed: int,
    frames: list,
    energies: np.ndarray,
    bins: np.ndarray,
    args: argparse.Namespace,
) -> None:
    rng = np.random.default_rng(seed)

    out = args.output_root / split_id / "splits"
    if out.exists() and args.overwrite:
        shutil.rmtree(out.parent)
    out.mkdir(parents=True, exist_ok=True)

    n_total = len(frames)
    n_train = int(round(args.train_frac * n_total))
    n_valid = int(round(args.valid_frac * n_total))
    n_test = n_total - n_train - n_valid

    train_indices = []
    valid_indices = []
    test_indices = []

    bin_sizes = [int(np.sum(bins == b)) for b in range(args.n_bins)]
    train_counts = allocate_counts(bin_sizes, n_train, args.train_frac)
    valid_counts = allocate_counts(bin_sizes, n_valid, args.valid_frac)

    for b in range(args.n_bins):
        idx = np.where(bins == b)[0].copy()
        rng.shuffle(idx)

        n_b_train = train_counts[b]
        n_b_valid = valid_counts[b]

        train_indices.extend(idx[:n_b_train].tolist())
        valid_indices.extend(idx[n_b_train : n_b_train + n_b_valid].tolist())
        test_indices.extend(idx[n_b_train + n_b_valid :].tolist())

    train_indices = np.asarray(train_indices, dtype=int)
    valid_indices = np.asarray(valid_indices, dtype=int)
    test_indices = np.asarray(test_indices, dtype=int)

    rng.shuffle(train_indices)
    rng.shuffle(valid_indices)
    rng.shuffle(test_indices)

    if len(train_indices) != n_train:
        raise RuntimeError(f"{split_id}: expected {n_train} train, got {len(train_indices)}")
    if len(valid_indices) != n_valid:
        raise RuntimeError(f"{split_id}: expected {n_valid} valid, got {len(valid_indices)}")
    if len(test_indices) != n_test:
        raise RuntimeError(f"{split_id}: expected {n_test} test, got {len(test_indices)}")

    used = np.concatenate([train_indices, valid_indices, test_indices])
    if len(np.unique(used)) != n_total:
        raise RuntimeError(f"{split_id}: split indices are not a partition")

    write(out / "formamide_train.xyz", [frames[i] for i in train_indices], format="extxyz")
    write(out / "formamide_valid.xyz", [frames[i] for i in valid_indices], format="extxyz")
    write(out / "formamide_test.xyz", [frames[i] for i in test_indices], format="extxyz")

    np.savetxt(out / "formamide_train_indices.txt", train_indices, fmt="%d")
    np.savetxt(out / "formamide_valid_indices.txt", valid_indices, fmt="%d")
    np.savetxt(out / "formamide_test_indices.txt", test_indices, fmt="%d")

    rows = []
    for set_name, indices in [
        ("train", train_indices),
        ("valid", valid_indices),
        ("test", test_indices),
    ]:
        for local_index, frame_index in enumerate(indices.tolist()):
            rows.append(
                {
                    "split_id": split_id,
                    "seed": seed,
                    "set": set_name,
                    "local_index": local_index,
                    "frame_index": int(frame_index),
                    "interaction_energy": float(energies[frame_index]),
                    "energy_bin": int(bins[frame_index]),
                }
            )

    with (out / "split_manifest.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "split_id",
                "seed",
                "set",
                "local_index",
                "frame_index",
                "interaction_energy",
                "energy_bin",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "split_id": split_id,
        "seed": seed,
        "n_total": n_total,
        "n_train": int(len(train_indices)),
        "n_valid": int(len(valid_indices)),
        "n_test": int(len(test_indices)),
        "train_frac": args.train_frac,
        "valid_frac": args.valid_frac,
        "test_frac": 1.0 - args.train_frac - args.valid_frac,
        "n_bins": int(args.n_bins),
        "stratification": "interaction_energy_rank_quantile_bins",
        "energy_min": float(np.min(energies)),
        "energy_max": float(np.max(energies)),
        "energy_mean": float(np.mean(energies)),
        "energy_std": float(np.std(energies)),
    }
    with (out / "split_summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)


def main() -> None:
    args = parse_args()

    if args.n_splits < 1:
        raise ValueError("--n-splits must be at least 1")
    if args.n_bins < 2:
        raise ValueError("--n-bins must be at least 2")
    if args.train_frac <= 0.0 or args.valid_frac <= 0.0:
        raise ValueError("train and validation fractions must be positive")
    if args.train_frac + args.valid_frac >= 1.0:
        raise ValueError("train + validation fractions must be < 1")

    frames = read(args.input_cg, ":")
    if not frames:
        raise RuntimeError(f"No frames read from {args.input_cg}")
    for frame in frames:
        frame.info['n_molecules'] = len(frame)
        frame.arrays['c_diameter\[1\]'] = frame.arrays.pop('c_diameter1')
        frame.arrays['c_diameter\[2\]'] = frame.arrays.pop('c_diameter2')
        frame.arrays['c_diameter\[3\]'] = frame.arrays.pop('c_diameter3')

    energies = np.asarray([interaction_energy(frame) for frame in frames], dtype=float)
    bins = rank_bins(energies, args.n_bins)

    args.output_root.mkdir(parents=True, exist_ok=True)

    for split_number in range(0, args.n_splits):
        split_id = f"split_{split_number:02d}"
        seed = args.seed_base + split_number
        write_replicate_split(
            split_id=split_id,
            seed=seed,
            frames=frames,
            energies=energies,
            bins=bins,
            args=args,
        )

    print(f"Wrote {args.n_splits} split replicates under {args.output_root.resolve()}")


if __name__ == "__main__":
    main()
