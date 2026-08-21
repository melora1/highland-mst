#!/usr/bin/env python3
"""Single entry point for the condensed revision codebase."""

from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

from analysis import (
    analyze_events,
    analyze_gradient,
    ensemble_adaptive_summary,
    ensemble_guard_gap_summary,
    ensemble_weight_closure_summary,
    ensemble_roi_split_summary,
    ensemble_roi_spill_summary,
    ensemble_artifact_summary,
    paired_seed_summary,
    refresh_gradient_summary,
    refresh_image_summaries,
    run_theory,
)
from plots import plot_gradient, plot_images, plot_theory
from simulation import (
    load_events,
    save_events,
    simulate_equal_exposure,
    simulate_gradient_exposure,
)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("theory")
    p.add_argument("--out", default="out/theory")

    p = sub.add_parser("simulate")
    p.add_argument("--n-per-setting", type=int, default=500_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="out/equal")

    p = sub.add_parser("gradient")
    p.add_argument("--n-per-cell", type=int, default=20_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="out/gradient")

    p = sub.add_parser("analyze")
    p.add_argument("file")
    p.add_argument("--out", required=True)
    p.add_argument("--gradient", action="store_true")

    p = sub.add_parser("postprocess")
    p.add_argument("outdirs", nargs="+", help="existing result directories containing images.npz")

    p = sub.add_parser("paired")
    p.add_argument(
        "metrics",
        nargs="+",
        help="metrics.csv files from matched seeds; shell globs are allowed",
    )
    p.add_argument("--out", default="out/paired_seed_summary.csv")

    p = sub.add_parser("all")
    p.add_argument("--n-per-setting", type=int, default=500_000)
    p.add_argument("--n-per-cell", type=int, default=20_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="out")

    a = ap.parse_args()
    if a.cmd == "theory":
        run_theory(a.out)
        plot_theory(a.out)
        return

    if a.cmd == "simulate":
        out = Path(a.out)
        out.mkdir(parents=True, exist_ok=True)
        df, cache = simulate_equal_exposure(a.n_per_setting, a.seed)
        actual = save_events(df, out / "events.parquet")
        print("events:", actual)
        analyze_events(df, out, cache=cache)
        plot_images(out)
        return

    if a.cmd == "gradient":
        out = Path(a.out)
        out.mkdir(parents=True, exist_ok=True)
        df, cache = simulate_gradient_exposure(a.n_per_cell, a.seed)
        actual = save_events(df, out / "events.parquet")
        print("events:", actual)
        analyze_gradient(df, out, cache=cache)
        plot_gradient(out)
        return

    if a.cmd == "analyze":
        df = load_events(a.file)
        if a.gradient:
            analyze_gradient(df, a.out)
        else:
            analyze_events(df, a.out)
        return

    if a.cmd == "postprocess":
        for d in a.outdirs:
            d = Path(d)
            if (d / "images.npz").exists():
                row = refresh_image_summaries(d)
                print(d, row)
            elif (d / "gradient_maps.npz").exists():
                print(refresh_gradient_summary(d).to_string(index=False))
            else:
                raise FileNotFoundError(f"{d}: no images.npz or gradient_maps.npz")
        return

    if a.cmd == "paired":
        out = Path(a.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        summary = paired_seed_summary(a.metrics, out_csv=out)
        artifact_files = [Path(f).with_name("artifact_summary.csv") for f in a.metrics]
        artifact_files = [f for f in artifact_files if f.exists()]
        if artifact_files:
            ensemble_artifact_summary(artifact_files, out.parent / "paired_artifact_summary.csv")
        adaptive_files = [Path(f).with_name("adaptive_retention.csv") for f in a.metrics]
        adaptive_files = [f for f in adaptive_files if f.exists()]
        if adaptive_files:
            ensemble_adaptive_summary(adaptive_files, out.parent / "paired_adaptive_retention.csv")
        for sibling, fn, target in (
            ("roi_guard_gap_sensitivity.csv", ensemble_guard_gap_summary, "paired_roi_guard_gap.csv"),
            ("weight_closure_by_momentum.csv", ensemble_weight_closure_summary, "paired_weight_closure.csv"),
            ("roi_split_half_metrics.csv", ensemble_roi_split_summary, "paired_roi_split_half.csv"),
            ("roi_spill.csv", ensemble_roi_spill_summary, "paired_roi_spill.csv"),
        ):
            files = [Path(f).with_name(sibling) for f in a.metrics]
            files = [f for f in files if f.exists()]
            if files:
                fn(files, out.parent / target)
        print(summary.to_string(index=False))
        return

    if a.cmd == "all":
        root = Path(a.out)
        root.mkdir(parents=True, exist_ok=True)
        t = root / "theory"
        run_theory(t)
        plot_theory(t)
        e = root / "equal"
        e.mkdir(exist_ok=True)
        df, cache = simulate_equal_exposure(a.n_per_setting, a.seed)
        save_events(df, e / "events.parquet")
        analyze_events(df, e, cache=cache)
        plot_images(e)
        g = root / "gradient"
        g.mkdir(exist_ok=True)
        dg, cacheg = simulate_gradient_exposure(a.n_per_cell, a.seed)
        save_events(dg, g / "events.parquet")
        analyze_gradient(dg, g, cache=cacheg)
        plot_gradient(g)


if __name__ == "__main__":
    main()
