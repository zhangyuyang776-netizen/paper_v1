from __future__ import annotations

"""Outer corrector for one outer-iteration update within a time step.

This module updates the next outer iterate from the current fixed-point
iterate and the inner-solve physical interface velocity. It does not perform
prediction, inner solves, convergence checks, step acceptance, grid builds, or
geometry commits.
"""

from dataclasses import dataclass
from math import isfinite

from .nonlinear_types import CorrectorResult


@dataclass(slots=True, kw_only=True)
class CorrectorConfigView:
    """Normalized outer-corrector configuration from ``cfg.outer_stepper``."""

    corrector_mode: str = "trapezoidal_fixed_point"
    omega_a: float = 1.0
    omega_v: float = 1.0


def _normalize_corrector_mode(mode: str) -> str:
    """Normalize and validate the supported corrector mode."""

    text = str(mode).strip().lower()
    if text != "trapezoidal_fixed_point":
        raise ValueError(f"unsupported corrector_mode: {mode!r}")
    return text


def _validate_relaxation_factor(name: str, value: float) -> float:
    """Validate one under-relaxation factor."""

    if not isfinite(float(value)):
        raise ValueError(f"{name} must be finite")
    value_f = float(value)
    if value_f <= 0.0 or value_f > 1.0:
        raise ValueError(f"{name} must satisfy 0 < {name} <= 1")
    return value_f


def build_corrector_config_view(cfg: object) -> CorrectorConfigView:
    """Build a normalized corrector config view from runtime configuration."""

    outer = getattr(cfg, "outer_stepper", None)
    mode = getattr(outer, "corrector_mode", "trapezoidal_fixed_point")
    omega_a = getattr(outer, "omega_a", 1.0)
    omega_v = getattr(outer, "omega_v", 1.0)
    return CorrectorConfigView(
        corrector_mode=_normalize_corrector_mode(mode),
        omega_a=_validate_relaxation_factor("omega_a", float(omega_a)),
        omega_v=_validate_relaxation_factor("omega_v", float(omega_v)),
    )


def compute_corrector_target(
    *,
    dt: float,
    a_old: float,
    dot_a_old: float,
    dot_a_phys: float,
) -> float:
    """Compute the trapezoidal target anchored to accepted old-time values."""

    return float(a_old) + 0.5 * float(dt) * (float(dot_a_old) + float(dot_a_phys))


def apply_corrector_relaxation(
    *,
    a_iter: float,
    dot_a_iter: float,
    a_target: float,
    dot_a_phys: float,
    omega_a: float,
    omega_v: float,
) -> tuple[float, float]:
    """Apply under-relaxation to the corrector target and physical interface speed."""

    a_new = (1.0 - float(omega_a)) * float(a_iter) + float(omega_a) * float(a_target)
    dot_a_new = (1.0 - float(omega_v)) * float(dot_a_iter) + float(omega_v) * float(dot_a_phys)
    return float(a_new), float(dot_a_new)


def compute_outer_corrector(
    *,
    cfg: object,
    dt: float,
    a_old: float,
    dot_a_old: float,
    a_iter: float,
    dot_a_iter: float,
    dot_a_phys: float,
    outer_iter_index: int | None = None,
) -> CorrectorResult:
    """Compute one outer corrector update from accepted old-time anchors."""

    cfg_corr = build_corrector_config_view(cfg)

    dt_f = float(dt)
    a_old_f = float(a_old)
    dot_a_old_f = float(dot_a_old)
    a_iter_f = float(a_iter)
    dot_a_iter_f = float(dot_a_iter)
    dot_a_phys_f = float(dot_a_phys)

    for name, value in (
        ("dt", dt_f),
        ("a_old", a_old_f),
        ("dot_a_old", dot_a_old_f),
        ("a_iter", a_iter_f),
        ("dot_a_iter", dot_a_iter_f),
        ("dot_a_phys", dot_a_phys_f),
    ):
        if not isfinite(value):
            raise ValueError(f"{name} must be finite")

    if dt_f <= 0.0:
        raise ValueError("corrector requires dt > 0")
    if a_old_f <= 0.0:
        raise ValueError("corrector requires a_old > 0")
    if a_iter_f <= 0.0:
        raise ValueError("corrector requires a_iter > 0")

    a_target = compute_corrector_target(
        dt=dt_f,
        a_old=a_old_f,
        dot_a_old=dot_a_old_f,
        dot_a_phys=dot_a_phys_f,
    )
    if not isfinite(a_target) or a_target <= 0.0:
        raise ValueError("corrector produced invalid a_target")

    a_new, dot_a_new = apply_corrector_relaxation(
        a_iter=a_iter_f,
        dot_a_iter=dot_a_iter_f,
        a_target=a_target,
        dot_a_phys=dot_a_phys_f,
        omega_a=cfg_corr.omega_a,
        omega_v=cfg_corr.omega_v,
    )
    if not isfinite(a_new) or a_new <= 0.0:
        raise ValueError("corrector produced invalid a_new")
    if not isfinite(dot_a_new):
        raise ValueError("corrector produced invalid dot_a_new")

    eps_dot_a = abs(dot_a_phys_f - dot_a_new)
    relaxed = not (cfg_corr.omega_a == 1.0 and cfg_corr.omega_v == 1.0)
    relaxation_factor = cfg_corr.omega_a if cfg_corr.omega_a == cfg_corr.omega_v else None

    return CorrectorResult(
        a_new=float(a_new),
        dot_a_new=float(dot_a_new),
        eps_dot_a=float(eps_dot_a),
        relaxed=relaxed,
        relaxation_factor=relaxation_factor,
        diagnostics={
            "corrector_mode": cfg_corr.corrector_mode,
            "outer_iter_index": outer_iter_index,
            "omega_a": cfg_corr.omega_a,
            "omega_v": cfg_corr.omega_v,
            "a_old": a_old_f,
            "dot_a_old": dot_a_old_f,
            "a_iter": a_iter_f,
            "dot_a_iter": dot_a_iter_f,
            "dot_a_phys": dot_a_phys_f,
            "a_target": float(a_target),
        },
    )


__all__ = [
    "CorrectorConfigView",
    "_normalize_corrector_mode",
    "_validate_relaxation_factor",
    "apply_corrector_relaxation",
    "build_corrector_config_view",
    "compute_corrector_target",
    "compute_outer_corrector",
]
