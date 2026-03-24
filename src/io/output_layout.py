from __future__ import annotations

"""Filesystem layout helpers for one simulation run and its failure artifacts.

This module defines the canonical path contract for:
- one normal run directory
- accepted-step spatial snapshots
- one failed step-attempt directory
- runtime text log / structured event stream
- optional rank sidecar artifacts written during failures
"""

from dataclasses import dataclass
from pathlib import Path
import re


@dataclass(frozen=True, slots=True)
class OutputLayout:
    """Resolved output paths for one run."""

    case_id: str
    run_id: str

    root: Path

    # Runtime logs
    run_log: Path
    run_events_jsonl: Path

    # Static run metadata
    config_echo_yaml: Path
    metadata_json: Path
    run_summary_json: Path

    # Accepted-step tabular outputs
    step_diag_csv: Path
    scalars_csv: Path
    interface_diag_csv: Path

    # Spatial outputs
    spatial_root: Path
    mapping_json: Path
    steps_dir: Path

    # Failure artifacts root
    failure_root: Path


@dataclass(frozen=True, slots=True)
class FailureLayout:
    """Resolved output paths for one failed step attempt."""

    root: Path

    # Main structured failure report
    report_json: Path

    # State snapshots around the failed attempt
    state_last_good_npz: Path
    state_attempt_input_npz: Path
    state_attempt_final_npz: Path

    # Spatial mapping snapshot used to interpret the failed state
    mapping_json: Path

    # Log tails copied into this failure directory for quick diagnosis
    run_log_tail_txt: Path
    run_events_tail_jsonl: Path


_VALID_TOKEN_RE = re.compile(r"[^A-Za-z0-9_-]+")
_MULTI_UNDERSCORE_RE = re.compile(r"_+")


def _normalize_token(raw: str, *, field_name: str) -> str:
    text = str(raw).strip()
    if not text:
        raise ValueError(f"{field_name} must be non-empty")
    text = text.replace(" ", "_")
    text = _VALID_TOKEN_RE.sub("_", text)
    text = _MULTI_UNDERSCORE_RE.sub("_", text).strip("_")
    if not text:
        raise ValueError(f"sanitized {field_name} is empty")
    return text


def _sanitize_case_id(case_id: str) -> str:
    """Normalize cfg case identifier into a safe directory token."""

    return _normalize_token(case_id, field_name="case_id")


def _sanitize_run_id(run_id: str) -> str:
    """Validate and normalize a run identifier into a safe directory token."""

    raw = str(run_id).strip()
    if not raw:
        raise ValueError("run_id must be non-empty")
    if "/" in raw or "\\" in raw or ".." in raw:
        raise ValueError(f"invalid run_id: {run_id!r}")
    return _normalize_token(raw, field_name="run_id")


def _extract_case_id(cfg) -> str:
    case_section = getattr(cfg, "case", None)
    if case_section is None:
        raise ValueError("cfg.case is required")
    if getattr(case_section, "case_id", None) is not None:
        return _sanitize_case_id(case_section.case_id)
    if getattr(case_section, "id", None) is not None:
        return _sanitize_case_id(case_section.id)
    raise ValueError("cfg.case.case_id or cfg.case.id is required")


def resolve_run_root(cfg, *, run_id: str) -> Path:
    """Resolve the root directory for one run without creating it."""

    case_id = _extract_case_id(cfg)
    rid = _sanitize_run_id(run_id)
    paths = getattr(cfg, "paths", None)
    output_root_value = getattr(paths, "output_root", None)
    if output_root_value is None:
        raise ValueError("cfg.paths.output_root is required")
    output_root = Path(output_root_value).expanduser()
    return output_root / case_id / rid


def build_output_layout(cfg, *, run_id: str) -> OutputLayout:
    """Build the canonical output layout for one run."""

    case_id = _extract_case_id(cfg)
    rid = _sanitize_run_id(run_id)
    root = resolve_run_root(cfg, run_id=rid)
    spatial_root = root / "3D_out"
    steps_dir = spatial_root / "steps"
    return OutputLayout(
        case_id=case_id,
        run_id=rid,
        root=root,
        run_log=root / "run.log",
        run_events_jsonl=root / "run.events.jsonl",
        config_echo_yaml=root / "config.echo.yaml",
        metadata_json=root / "metadata.json",
        run_summary_json=root / "run_summary.json",
        step_diag_csv=root / "step_diag.csv",
        scalars_csv=root / "scalars.csv",
        interface_diag_csv=root / "interface_diag.csv",
        spatial_root=spatial_root,
        mapping_json=spatial_root / "mapping.json",
        steps_dir=steps_dir,
        failure_root=root,
    )


def ensure_output_dirs(layout: OutputLayout) -> None:
    """Create the stable run output directories.

    This function only creates directories, not files.
    """

    layout.root.mkdir(parents=True, exist_ok=True)
    layout.spatial_root.mkdir(parents=True, exist_ok=True)
    layout.steps_dir.mkdir(parents=True, exist_ok=True)


def step_snapshot_path(layout: OutputLayout, *, step_id: int, t: float) -> Path:
    """Return the canonical spatial snapshot path for one accepted step."""

    if int(step_id) < 0:
        raise ValueError("step_id must be >= 0")
    return layout.steps_dir / f"step_{int(step_id):06d}_time_{float(t):.6e}.npz"


def failure_dir(layout: OutputLayout, *, step_id: int, retry_id: int) -> Path:
    """Return the canonical directory for one failed step attempt."""

    if int(step_id) < 0:
        raise ValueError("step_id must be >= 0")
    if int(retry_id) < 0:
        raise ValueError("retry_id must be >= 0")
    return layout.failure_root / f"failed_step_{int(step_id):06d}_retry_{int(retry_id):02d}"


def build_failure_layout(layout: OutputLayout, *, step_id: int, retry_id: int) -> FailureLayout:
    """Build the canonical file layout for one failed step attempt."""

    root = failure_dir(layout, step_id=step_id, retry_id=retry_id)
    return FailureLayout(
        root=root,
        report_json=root / "failure_report.json",
        state_last_good_npz=root / "state_last_good.npz",
        state_attempt_input_npz=root / "state_attempt_input.npz",
        state_attempt_final_npz=root / "state_attempt_final.npz",
        mapping_json=root / "mapping.json",
        run_log_tail_txt=root / "run_log_tail.txt",
        run_events_tail_jsonl=root / "run_events_tail.jsonl",
    )


def ensure_failure_dir(layout: FailureLayout) -> None:
    """Create one failure-attempt directory."""

    layout.root.mkdir(parents=True, exist_ok=True)


def failure_rank_meta_path(layout: FailureLayout, *, rank: int) -> Path:
    """Return the canonical per-rank metadata json path for one failure directory."""

    if int(rank) < 0:
        raise ValueError("rank must be >= 0")
    return layout.root / f"rank_{int(rank):03d}_meta.json"


def failure_rank_snes_last_x_csv_path(layout: FailureLayout, *, rank: int) -> Path:
    """Return the canonical per-rank SNES last-x csv path for one failure directory."""

    if int(rank) < 0:
        raise ValueError("rank must be >= 0")
    return layout.root / f"rank_{int(rank):03d}_snes_last_x.csv"


__all__ = [
    "FailureLayout",
    "OutputLayout",
    "_sanitize_case_id",
    "_sanitize_run_id",
    "build_failure_layout",
    "build_output_layout",
    "ensure_failure_dir",
    "ensure_output_dirs",
    "failure_dir",
    "failure_rank_meta_path",
    "failure_rank_snes_last_x_csv_path",
    "resolve_run_root",
    "step_snapshot_path",
]
