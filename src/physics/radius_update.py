from __future__ import annotations

"""Outer-level physical interface-speed extraction for paper_v1.

This module is not part of inner residual assembly. It only extracts the
physical interface radial speed from the converged inner interface truth for
use by the outer corrector.

Forbidden here:
- updating a or dot_a_frozen
- applying correctors or under-relaxation
- timestep accept/reject logic
- recomputing u_l_if_abs
- recomputing rho_s_l or mpp
- reviving the old radius_eq.py inner-implicit radius-equation route
"""

from dataclasses import dataclass
from typing import Any

import numpy as np

from physics.interface_face import InterfaceFacePackage
from physics.velocity_recovery import VelocityRecoveryPackage


class RadiusUpdateError(ValueError):
    """Raised when radius-update inputs are inconsistent."""


@dataclass(frozen=True)
class RadiusUpdatePackage:
    dot_a_phys: float
    dot_a_frozen: float
    eps_dot_a: float

    u_l_if_abs: float
    mpp: float
    rho_s_l: float

    diagnostics: dict[str, Any]


def _validate_finite_scalar(name: str, value: float) -> float:
    scalar = float(value)
    if not np.isfinite(scalar):
        raise RadiusUpdateError(f"{name} must be finite")
    return scalar


def _validate_positive_scalar(name: str, value: float) -> float:
    scalar = _validate_finite_scalar(name, value)
    if scalar <= 0.0:
        raise RadiusUpdateError(f"{name} must be > 0")
    return scalar


def _compute_dot_a_phys(
    u_l_if_abs: float,
    mpp: float,
    rho_s_l: float,
) -> float:
    return float(u_l_if_abs + mpp / rho_s_l)


def _compute_outer_velocity_mismatch(
    dot_a_phys: float,
    dot_a_frozen: float,
    eps_dot_a_floor: float,
) -> float:
    denom = max(abs(dot_a_phys), float(eps_dot_a_floor))
    return float(abs(dot_a_phys - dot_a_frozen) / denom)


def _build_radius_update_diagnostics(
    *,
    dot_a_phys: float,
    dot_a_frozen: float,
    eps_dot_a: float,
    u_l_if_abs: float,
    mpp: float,
    rho_s_l: float,
) -> dict[str, Any]:
    return {
        "dot_a_phys": float(dot_a_phys),
        "dot_a_frozen": float(dot_a_frozen),
        "eps_dot_a": float(eps_dot_a),
        "u_l_if_abs": float(u_l_if_abs),
        "mpp": float(mpp),
        "rho_s_l": float(rho_s_l),
    }


def build_radius_update_package(
    iface_pkg: InterfaceFacePackage,
    vel_pkg: VelocityRecoveryPackage,
    *,
    eps_dot_a_floor: float = 1.0e-12,
) -> RadiusUpdatePackage:
    """Extract dot_a_phys and outer mismatch from converged interface truth."""

    rho_s_l = _validate_positive_scalar("iface_pkg.rho_s_l", iface_pkg.rho_s_l)
    mpp = _validate_finite_scalar("iface_pkg.mpp", iface_pkg.mpp)
    dot_a_frozen = _validate_finite_scalar("iface_pkg.dot_a_frozen", iface_pkg.dot_a_frozen)
    u_l_if_abs = _validate_finite_scalar("vel_pkg.u_l_if_abs", vel_pkg.u_l_if_abs)
    eps_floor = _validate_positive_scalar("eps_dot_a_floor", eps_dot_a_floor)

    dot_a_phys = _compute_dot_a_phys(u_l_if_abs, mpp, rho_s_l)
    eps_dot_a = _compute_outer_velocity_mismatch(dot_a_phys, dot_a_frozen, eps_floor)
    diagnostics = _build_radius_update_diagnostics(
        dot_a_phys=dot_a_phys,
        dot_a_frozen=dot_a_frozen,
        eps_dot_a=eps_dot_a,
        u_l_if_abs=u_l_if_abs,
        mpp=mpp,
        rho_s_l=rho_s_l,
    )

    return RadiusUpdatePackage(
        dot_a_phys=float(dot_a_phys),
        dot_a_frozen=float(dot_a_frozen),
        eps_dot_a=float(eps_dot_a),
        u_l_if_abs=float(u_l_if_abs),
        mpp=float(mpp),
        rho_s_l=float(rho_s_l),
        diagnostics=diagnostics,
    )
