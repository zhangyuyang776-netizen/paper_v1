from __future__ import annotations

"""Outer convergence check for fixed-point coupling between outer and inner solves.

This module only evaluates the outer convergence criterion. It does not run
predictor/corrector steps, inner solves, acceptance logic, or time-step
control.
"""

from dataclasses import dataclass
from math import isfinite

from .nonlinear_types import OuterConvergenceResult


@dataclass(slots=True, kw_only=True)
class OuterConvergenceConfigView:
    """Normalized outer-convergence configuration from ``cfg.outer_stepper``."""

    convergence_mode: str = "eps_dot_a"
    eps_dot_a_tol: float = 1.0e-5
    eps_ref_dot_a: float = 1.0e-12


def _normalize_convergence_mode(mode: str) -> str:
    """Normalize and validate the supported outer-convergence mode."""

    text = str(mode).strip().lower()
    if text != "eps_dot_a":
        raise ValueError(f"unsupported outer_convergence_mode: {mode!r}")
    return text


def _validate_positive_finite(name: str, value: float) -> float:
    """Validate a positive finite scalar configuration value."""

    value_f = float(value)
    if not isfinite(value_f):
        raise ValueError(f"{name} must be finite")
    if value_f <= 0.0:
        raise ValueError(f"{name} must be > 0")
    return value_f


def build_outer_convergence_config_view(cfg: object) -> OuterConvergenceConfigView:
    """Build a normalized outer-convergence config view from runtime config."""

    outer = getattr(cfg, "outer_stepper", None)
    mode = getattr(outer, "outer_convergence_mode", "eps_dot_a")
    tol = getattr(outer, "outer_convergence_tol", 1.0e-5)
    eps_ref = getattr(outer, "eps_ref_dot_a", 1.0e-12)
    return OuterConvergenceConfigView(
        convergence_mode=_normalize_convergence_mode(mode),
        eps_dot_a_tol=_validate_positive_finite("outer_convergence_tol", float(tol)),
        eps_ref_dot_a=_validate_positive_finite("eps_ref_dot_a", float(eps_ref)),
    )


def compute_eps_dot_a(
    *,
    dot_a_iter: float,
    dot_a_phys: float,
    eps_ref_dot_a: float,
) -> float:
    """Compute the normalized outer velocity mismatch."""

    denom = max(abs(float(dot_a_phys)), float(eps_ref_dot_a))
    return abs(float(dot_a_phys) - float(dot_a_iter)) / denom


def evaluate_outer_convergence(
    *,
    cfg: object,
    dot_a_iter: float,
    dot_a_phys: float,
    outer_iter_index: int | None = None,
    eps_dot_a_prev: float | None = None,
) -> OuterConvergenceResult:
    """Evaluate the frozen outer convergence criterion from current-guess mismatch."""

    cfg_conv = build_outer_convergence_config_view(cfg)

    dot_a_iter_f = float(dot_a_iter)
    dot_a_phys_f = float(dot_a_phys)
    if not isfinite(dot_a_iter_f):
        raise ValueError("dot_a_iter must be finite")
    if not isfinite(dot_a_phys_f):
        raise ValueError("dot_a_phys must be finite")
    if eps_dot_a_prev is not None and not isfinite(float(eps_dot_a_prev)):
        raise ValueError("eps_dot_a_prev must be finite when provided")

    eps_dot_a = compute_eps_dot_a(
        dot_a_iter=dot_a_iter_f,
        dot_a_phys=dot_a_phys_f,
        eps_ref_dot_a=cfg_conv.eps_ref_dot_a,
    )
    converged = eps_dot_a < cfg_conv.eps_dot_a_tol
    nonmonotonic = eps_dot_a_prev is not None and float(eps_dot_a) > float(eps_dot_a_prev)
    denominator = max(abs(dot_a_phys_f), float(cfg_conv.eps_ref_dot_a))

    return OuterConvergenceResult(
        converged=bool(converged),
        eps_dot_a=float(eps_dot_a),
        tolerance=float(cfg_conv.eps_dot_a_tol),
        iteration_index=int(outer_iter_index or 0),
        nonmonotonic_flag=bool(nonmonotonic),
        diagnostics={
            "convergence_mode": cfg_conv.convergence_mode,
            "dot_a_iter": dot_a_iter_f,
            "dot_a_phys": dot_a_phys_f,
            "eps_ref_dot_a": float(cfg_conv.eps_ref_dot_a),
            "eps_dot_a_prev": None if eps_dot_a_prev is None else float(eps_dot_a_prev),
            "relative_denominator": float(denominator),
        },
    )


__all__ = [
    "OuterConvergenceConfigView",
    "_normalize_convergence_mode",
    "_validate_positive_finite",
    "build_outer_convergence_config_view",
    "compute_eps_dot_a",
    "evaluate_outer_convergence",
]
