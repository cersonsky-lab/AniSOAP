#!/usr/bin/env python
"""Benchmark AniSOAP Gaussian moment generation schemes.

Run from the AniSOAP repo root, or from any environment where ``anisoap`` is
importable:

    python benchmark_moment_generation.py --device cpu --dtype float64
    python benchmark_moment_generation.py --device cuda --dtype float32

This compares:
  1. legacy/single ``compute_moments`` called once per Gaussian;
  2. batched ``compute_moments_batched``;
  3. optionally ``torch.compile(compute_moments_batched)``.

The benchmark also checks numerical agreement between legacy and batched output
for a small subset of Gaussians.
"""

from __future__ import annotations

import argparse
import gc
import math
import statistics
import time
from typing import Callable, Optional, Tuple

import torch

from anisoap.representations.ellipsoidal_density_projection import compute_moments

try:
    from anisoap.representations.ellipsoidal_density_projection import (
        compute_moments_batched,
    )
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "Could not import compute_moments_batched. Apply the batched-moments "
        "patch first."
    ) from exc


def parse_dtype(name: str) -> torch.dtype:
    if name == "float32":
        return torch.float32
    if name == "float64":
        return torch.float64
    raise ValueError(f"Unsupported dtype: {name}")


def make_spd_precision_matrices(
    n: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
) -> torch.Tensor:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)

    raw = torch.randn((n, 3, 3), generator=gen, dtype=dtype).to(device)
    eye = torch.eye(3, device=device, dtype=dtype).expand(n, 3, 3)

    # SPD with moderate condition numbers; this avoids benchmarking numerical
    # pathologies instead of moment generation.
    A = raw @ raw.transpose(-1, -2) + 0.75 * eye
    return 0.5 * (A + A.transpose(-1, -2))


def make_centers(
    n: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
) -> torch.Tensor:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed + 1)
    return torch.randn((n, 3), generator=gen, dtype=dtype).to(device) * 0.5


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def time_callable(
    fn: Callable[[], object],
    *,
    device: torch.device,
    warmup: int,
    repeats: int,
) -> Tuple[float, float, float]:
    for _ in range(warmup):
        fn()
    sync(device)

    times = []
    for _ in range(repeats):
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        sync(device)
        t0 = time.perf_counter()
        fn()
        sync(device)
        times.append(time.perf_counter() - t0)

    return min(times), statistics.median(times), max(times)


def legacy_loop(A: torch.Tensor, centers: torch.Tensor, maxdeg: int) -> torch.Tensor:
    # compute_moments returns a dense cube per Gaussian. This intentionally uses
    # the old public interface to measure the current per-Gaussian overhead.
    chunks = []
    for i in range(A.shape[0]):
        chunks.append(compute_moments(A[i], centers[i], maxdeg).reshape(1, -1))
    return torch.cat(chunks, dim=0)


def legacy_loop_valid_only(
    A: torch.Tensor,
    centers: torch.Tensor,
    maxdeg: int,
    exponents: torch.Tensor,
) -> torch.Tensor:
    cubes = legacy_loop(A, centers, maxdeg)
    side = maxdeg + 1
    linear = exponents[:, 0] * side * side + exponents[:, 1] * side + exponents[:, 2]
    return cubes[:, linear]


def check_correctness(
    A: torch.Tensor,
    centers: torch.Tensor,
    maxdeg: int,
    n_check: int,
    rtol: float,
    atol: float,
) -> None:
    n_check = min(n_check, A.shape[0])
    A_small = A[:n_check]
    c_small = centers[:n_check]

    batched, exponents = compute_moments_batched(A_small, c_small, maxdeg)
    legacy = legacy_loop_valid_only(A_small, c_small, maxdeg, exponents)

    torch.testing.assert_close(batched, legacy, rtol=rtol, atol=atol)


def bytes_for(t: torch.Tensor) -> int:
    return t.numel() * t.element_size()


def maybe_compile(fn: Callable) -> Optional[Callable]:
    compile_fn = getattr(torch, "compile", None)
    if compile_fn is None:
        return None
    try:
        return compile_fn(fn, fullgraph=False)
    except Exception:
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=512, help="number of Gaussians")
    parser.add_argument("--maxdeg", type=int, default=8)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--dtype", type=str, default="float64", choices=["float32", "float64"])
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--check", type=int, default=8, help="number of entries used for correctness check")
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--skip-legacy", action="store_true", help="skip slow legacy loop timing")
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype = parse_dtype(args.dtype)

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")

    A = make_spd_precision_matrices(args.n, device=device, dtype=dtype, seed=args.seed)
    centers = make_centers(args.n, device=device, dtype=dtype, seed=args.seed)

    print("AniSOAP moment generation benchmark")
    print(f"  n        = {args.n}")
    print(f"  maxdeg   = {args.maxdeg}")
    print(f"  device   = {device}")
    print(f"  dtype    = {dtype}")
    print(f"  repeats  = {args.repeats}")
    print()

    # Correctness check uses valid monomials only.
    check_correctness(
        A,
        centers,
        args.maxdeg,
        n_check=args.check,
        rtol=5e-5 if dtype is torch.float32 else 1e-10,
        atol=5e-6 if dtype is torch.float32 else 1e-10,
    )
    print(f"correctness: batched matches legacy on first {min(args.check, args.n)} Gaussians")
    print()

    results = []

    batched_out, exponents = compute_moments_batched(A, centers, args.maxdeg)
    sync(device)
    n_valid = exponents.shape[0]
    dense_cube = (args.maxdeg + 1) ** 3
    print(f"valid monomials = {n_valid}")
    print(f"dense cube size = {dense_cube}")
    print(f"batched output memory = {bytes_for(batched_out) / 1024**2:.3f} MiB")
    print()

    def batched_fn() -> torch.Tensor:
        out, _ = compute_moments_batched(A, centers, args.maxdeg)
        return out

    t_min, t_med, t_max = time_callable(
        batched_fn, device=device, warmup=args.warmup, repeats=args.repeats
    )
    results.append(("batched", t_min, t_med, t_max))

    if not args.no_compile:
        compiled_batched = maybe_compile(compute_moments_batched)
        if compiled_batched is not None:
            def compiled_fn() -> torch.Tensor:
                out, _ = compiled_batched(A, centers, args.maxdeg)
                return out

            t_min, t_med, t_max = time_callable(
                compiled_fn, device=device, warmup=args.warmup, repeats=args.repeats
            )
            results.append(("batched torch.compile", t_min, t_med, t_max))
        else:
            print("torch.compile unavailable or failed to initialize; skipping compiled benchmark")
            print()

    if not args.skip_legacy:
        def legacy_fn() -> torch.Tensor:
            return legacy_loop_valid_only(A, centers, args.maxdeg, exponents)

        t_min, t_med, t_max = time_callable(
            legacy_fn, device=device, warmup=max(1, args.warmup // 2), repeats=args.repeats
        )
        results.append(("legacy loop", t_min, t_med, t_max))

    print("timings")
    print("  scheme                    min [ms]   median [ms]   max [ms]   speedup vs legacy")
    legacy_med = None
    for name, _, med, _ in results:
        if name == "legacy loop":
            legacy_med = med
            break

    for name, t_min, t_med, t_max in results:
        if legacy_med is None or name == "legacy loop":
            speedup = "--"
        else:
            speedup = f"{legacy_med / t_med:8.2f}x"
        print(
            f"  {name:<24} {1e3*t_min:9.3f} {1e3*t_med:12.3f} {1e3*t_max:9.3f}   {speedup}"
        )

    print()
    print("notes")
    print("  - legacy loop measures public compute_moments called once per Gaussian")
    print("  - batched output stores only valid total-degree monomials")
    print("  - use --skip-legacy for large n/maxdeg where the old loop is too slow")


if __name__ == "__main__":
    main()
