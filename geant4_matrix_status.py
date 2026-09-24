#!/usr/bin/env python3
"""Audit the 72-file Geant4 matrix and its accepted-exit gate."""

from __future__ import annotations

import argparse
from pathlib import Path
import re

import pandas as pd

from rev18_metadata import write_csv, write_text


COMPLETION = re.compile(r"\[RunAction\] wrote (\d+) / (\d+) primary exit angles")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", default="out/geant4/task8")
    parser.add_argument("--out", default="out/rev18/N_geant4")
    parser.add_argument("--min-accepted", type=int, default=1_300_000)
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    counter = 0
    for material in ("Cu", "Pb"):
        for momentum in ("1.0", "2.0", "3.5", "6.0"):
            for seed_index in range(1, 4):
                counter += 1
                for transport, offset in (
                    ("ftfp_bert", 1), ("ftfp_bert_wvi", 2), ("wvi_ss", 3)
                ):
                    seed = 510000 + counter*10 + offset
                    stem = f"{material}_t15.0_p{momentum}_{transport}_s{seed}"
                    raw = run_dir / "raw" / f"{stem}.txt"
                    log = run_dir / "logs" / f"{stem}.log"
                    match = None
                    if log.exists():
                        matches = COMPLETION.findall(log.read_text(errors="replace"))
                        match = matches[-1] if matches else None
                    rows.append(dict(
                        material=material, p_GeV=float(momentum),
                        seed_index=seed_index, transport=transport, seed=seed,
                        raw_file=str(raw), log_file=str(log),
                        raw_exists=raw.is_file(), log_exists=log.is_file(),
                        complete=match is not None,
                        accepted_exits=int(match[0]) if match else 0,
                        generated_primaries=int(match[1]) if match else 0,
                    ))
    seeds = pd.DataFrame(rows)
    meta = dict(
        generator="geant4_matrix_status.py", form_factor="mixed", floor="mixed",
        electron_cutoff="step", grid_resolution="72 raw transports", seed=None,
    )
    write_csv(seeds, out / "seed_matrix_status.csv", **meta)
    grouped = seeds.groupby(
        ["material", "p_GeV", "transport"], as_index=False
    ).agg(
        n_seeds=("seed_index", "size"),
        n_complete=("complete", "sum"),
        accepted_exits=("accepted_exits", "sum"),
        generated_primaries=("generated_primaries", "sum"),
    )
    grouped["pass_three_seed_matrix"] = grouped.n_complete == 3
    grouped["pass_min_accepted"] = grouped.accepted_exits >= int(args.min_accepted)
    grouped["pass"] = grouped.pass_three_seed_matrix & grouped.pass_min_accepted
    write_csv(grouped, out / "configuration_gate.csv", **meta)
    version_file = run_dir / "geant4_version.txt"
    version = version_file.read_text().strip() if version_file.exists() else "missing"
    summary = (
        f"Geant4 version: {version}\n"
        f"Complete seed transports: {int(seeds.complete.sum())}/{len(seeds)}\n"
        f"Passing configurations: {int(grouped['pass'].sum())}/{len(grouped)}\n"
        f"Minimum accepted exits: {int(grouped.accepted_exits.min())}\n"
        f"Required accepted exits: {int(args.min_accepted)}\n"
    )
    write_text(out / "matrix_gate_summary.txt", summary, **meta)
    if not bool(grouped["pass"].all()):
        raise SystemExit("Geant4 matrix gate failed; see configuration_gate.csv")
    print(summary, end="")


if __name__ == "__main__":
    main()
