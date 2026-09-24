"""Shared provenance sidecars for Rev. 18 publication outputs."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

from physics import FINITE_SIZE_KERNEL_VERSION
from reduced_cache import CACHE_SCHEMA_VERSION


def _git_revision() -> tuple[str, bool]:
    root = Path(__file__).resolve().parent
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    return commit, dirty


def metadata(
    *,
    form_factor: str = "mixed",
    floor: str | bool = "mixed",
    electron_cutoff: str = "step",
    grid_resolution=None,
    seed=None,
    generator: str,
    extra: dict | None = None,
) -> dict:
    commit, dirty = _git_revision()
    result = {
        "git_commit": commit,
        "git_worktree_dirty": dirty,
        "cache_version": CACHE_SCHEMA_VERSION,
        "kernel_identifier": (
            f"{FINITE_SIZE_KERNEL_VERSION};floor={str(floor).lower()};"
            f"electron_cutoff={electron_cutoff}"
        ),
        "form_factor": form_factor,
        "floor": floor,
        "electron_cutoff": electron_cutoff,
        "grid_resolution": grid_resolution if grid_resolution is not None else "n/a",
        "seed": seed,
        "generator": generator,
        "generation_timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    if extra:
        result.update(extra)
    return result


def attach_metadata(path, **kwargs) -> Path:
    """Write ``<output>.metadata.json`` next to one generated output."""
    output = Path(path)
    sidecar = Path(str(output) + ".metadata.json")
    sidecar.write_text(json.dumps(metadata(**kwargs), indent=2, sort_keys=True) + "\n")
    return sidecar


def write_text(path, content: str, **kwargs) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(content)
    attach_metadata(output, **kwargs)
    return output


def write_csv(frame, path, **kwargs) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    attach_metadata(output, **kwargs)
    return output


def attach_figure(path, **kwargs) -> Path:
    return attach_metadata(path, **kwargs)
