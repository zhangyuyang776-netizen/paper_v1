from __future__ import annotations

"""Outer predictor for the first outer iterate of each time step.

This module owns the step-initial predictor only. It does not build grids,
perform remap/recovery, run the inner solve, execute the outer corrector, or
participate in step acceptance.
"""

from dataclasses import dataclass
from math import isfinite

from .nonlinear_types import PredictorResult


@dataclass(slots=True, kw_only=True)
class PredictorConfigView:
    """Normalized predictor configuration extracted from ``cfg.outer_stepper``."""

    predictor_mode: str = "explicit_from_previous_dot_a"


def _normalize_predictor_mode(mode: str) -> str:
    """Normalize and validate the supported predictor mode."""

    text = str(mode).strip().lower()
    if text != "explicit_from_previous_dot_a":
        raise ValueError(f"unsupported predictor_mode: {mode!r}")
    return text


def build_predictor_config_view(cfg: object) -> PredictorConfigView:
    """Build a normalized predictor config view from runtime configuration."""

    outer = getattr(cfg, "outer_stepper", None)
    mode = getattr(outer, "predictor_mode", "explicit_from_previous_dot_a")
    return PredictorConfigView(
        predictor_mode=_normalize_predictor_mode(mode),
    )


def compute_outer_predictor(
    *,
    cfg: object,
    t_old: float,
    dt: float,
    a_old: float,
    dot_a_old: float,
    step_id: int | None = None,
) -> PredictorResult:
    """Compute the step-initial outer predictor from accepted old-time values."""

    cfg_pred = build_predictor_config_view(cfg)
    _ = cfg_pred

    t_old_f = float(t_old)
    dt_f = float(dt)
    a_old_f = float(a_old)
    dot_a_old_f = float(dot_a_old)

    if not isfinite(t_old_f):
        raise ValueError("predictor requires finite t_old")
    if not isfinite(dt_f) or dt_f <= 0.0:
        raise ValueError("predictor requires dt > 0")
    if not isfinite(a_old_f) or a_old_f <= 0.0:
        raise ValueError("predictor requires a_old > 0")
    if not isfinite(dot_a_old_f):
        raise ValueError("predictor requires finite dot_a_old")

    dot_a_pred = dot_a_old_f
    a_pred = a_old_f + dt_f * dot_a_pred
    if not isfinite(a_pred):
        raise ValueError("predictor produced non-finite a_pred")
    if a_pred <= 0.0:
        raise ValueError("predictor produced non-positive a_pred")

    first_step = step_id == 0
    return PredictorResult(
        a_pred=float(a_pred),
        dot_a_pred=float(dot_a_pred),
        first_step_special_case=bool(first_step),
        message="explicit Euler outer predictor",
        diagnostics={
            "predictor_mode": cfg_pred.predictor_mode,
            "t_old": t_old_f,
            "dt": dt_f,
            "a_old": a_old_f,
            "dot_a_old": dot_a_old_f,
        },
    )


__all__ = [
    "PredictorConfigView",
    "_normalize_predictor_mode",
    "build_predictor_config_view",
    "compute_outer_predictor",
]
