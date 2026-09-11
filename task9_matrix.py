#!/usr/bin/env python3
"""Build the remaining Task 9 cache matrix with one template preparation."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from reduced_cache import ReducedSamplerCache, build_reduced_cache, default_eta_grid
from simulation import load_events
from validation import _task9_empirical_templates


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("events")
    parser.add_argument("base_cache")
    parser.add_argument("support_dirs", nargs="+")
    parser.add_argument("--out", default="out/validation/task9")
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    base_path = Path(args.base_cache)
    base = ReducedSamplerCache.load(base_path)
    frames = []
    kink_counts = []
    for raw_dir in args.support_dirs:
        support_dir = Path(raw_dir)
        for geometry in ("target", "reference"):
            matches = sorted(
                support_dir.glob(f"support_unique_{geometry}_k*.parquet")
            )
            if len(matches) != 1:
                raise ValueError(
                    f"expected one compact {geometry} support file in {support_dir}"
                )
            n_kinks = int(matches[0].stem.rsplit("_k", 1)[1])
            q = pd.read_parquet(matches[0])
            q = q[
                q.B.notna() & q.rho.notna() & (q.rho > 0.0) & (q.s > 0.0)
            ].copy()
            q["geometry"] = geometry
            q["n_kinks"] = n_kinks
            frames.append(q)
            kink_counts.append(n_kinks)
    support = pd.concat(frames, ignore_index=True)
    events = load_events(args.events)
    print("preparing shared empirical templates", flush=True)
    templates = _task9_empirical_templates(
        events, support, base.B_grid, base.rho_grid
    )

    base_metadata_path = base_path.with_suffix(".metadata.csv")
    metadata = pd.read_csv(base_metadata_path)
    eta_max = float(metadata.eta_max.iloc[0])
    eta_grid = default_eta_grid(eta_max)
    for model, floor, label in (
        ("gaussian", False, "gaussian_floor_off"),
        ("uniform_sphere", True, "sphere_floor_on"),
        ("uniform_sphere", False, "sphere_floor_off"),
    ):
        stem = base_path.stem.replace("gaussian_floor_on", label)
        destination = out / f"{stem}.npz"
        if destination.exists():
            print("skipping existing", destination, flush=True)
            continue
        print("building", label, flush=True)
        cache = build_reduced_cache(
            base.B_grid,
            base.rho_grid,
            eta_grid,
            model,
            floor,
            templates=templates,
            workers=args.workers,
        )
        cache.save(destination)
        row = metadata.copy()
        row["ff_model"] = model
        row["floor"] = floor
        row["n_kinks"] = ";".join(map(str, sorted(set(kink_counts))))
        row.to_csv(destination.with_suffix(".metadata.csv"), index=False)
        print("wrote", destination, flush=True)


if __name__ == "__main__":
    main()
