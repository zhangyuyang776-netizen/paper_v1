from __future__ import annotations

"""Semantic interface-energy residual builder for paper_v1.

This module consumes the unique InterfaceFacePackage and translates its already
constructed interface energy quantities into a residual package. It must not
recompute interface heat fluxes, total energy fluxes, diffusive fluxes,
equilibrium states, or any material properties.

Forbidden here:
- recomputing q_l_s / q_g_s
- recomputing E_l_s / E_g_s
- calling physics.energy_flux again
- calling properties.equilibrium again
- adding any second mpp-related equation
- assembling directly into a global residual vector
"""

from dataclasses import dataclass
from typing import Any

import numpy as np

from physics.interface_face import InterfaceFacePackage


class InterfaceEnergyError(ValueError):
    """Raised when interface-energy residual inputs are inconsistent."""


@dataclass(frozen=True)
class InterfaceEnergyResidualPackage:
    energy_residual: float

    q_l_s: float
    q_g_s: float
    E_l_s: float
    E_g_s: float

    q_l_cond: float
    q_g_cond: float
    q_l_species_diff: float
    q_g_species_diff: float

    diagnostics: dict[str, Any]


def _validate_finite_scalar(name: str, value: float) -> float:
    scalar = float(value)
    if not np.isfinite(scalar):
        raise InterfaceEnergyError(f"{name} must be finite")
    return scalar


def _validate_interface_energy_inputs(iface_pkg: InterfaceFacePackage) -> None:
    _validate_finite_scalar("iface_pkg.E_l_s", iface_pkg.E_l_s)
    _validate_finite_scalar("iface_pkg.E_g_s", iface_pkg.E_g_s)
    _validate_finite_scalar("iface_pkg.q_l_s", iface_pkg.q_l_s)
    _validate_finite_scalar("iface_pkg.q_g_s", iface_pkg.q_g_s)
    _validate_finite_scalar("iface_pkg.q_l_cond", iface_pkg.q_l_cond)
    _validate_finite_scalar("iface_pkg.q_g_cond", iface_pkg.q_g_cond)
    _validate_finite_scalar("iface_pkg.q_l_species_diff", iface_pkg.q_l_species_diff)
    _validate_finite_scalar("iface_pkg.q_g_species_diff", iface_pkg.q_g_species_diff)


def _build_energy_residual(iface_pkg: InterfaceFacePackage) -> float:
    return float(iface_pkg.E_g_s - iface_pkg.E_l_s)


def _build_energy_diagnostics(
    iface_pkg: InterfaceFacePackage,
    energy_residual: float,
) -> dict[str, Any]:
    return {
        "energy_residual": float(energy_residual),
        "q_l_s": float(iface_pkg.q_l_s),
        "q_g_s": float(iface_pkg.q_g_s),
        "E_l_s": float(iface_pkg.E_l_s),
        "E_g_s": float(iface_pkg.E_g_s),
        "q_l_cond": float(iface_pkg.q_l_cond),
        "q_g_cond": float(iface_pkg.q_g_cond),
        "q_l_species_diff": float(iface_pkg.q_l_species_diff),
        "q_g_species_diff": float(iface_pkg.q_g_species_diff),
        "q_jump": float(iface_pkg.q_g_s - iface_pkg.q_l_s),
    }


def build_interface_energy_residual_package(
    iface_pkg: InterfaceFacePackage,
) -> InterfaceEnergyResidualPackage:
    """Translate InterfaceFacePackage energy data into a semantic residual package."""

    _validate_interface_energy_inputs(iface_pkg)
    energy_residual = _build_energy_residual(iface_pkg)
    diagnostics = _build_energy_diagnostics(iface_pkg, energy_residual)

    return InterfaceEnergyResidualPackage(
        energy_residual=float(energy_residual),
        q_l_s=float(iface_pkg.q_l_s),
        q_g_s=float(iface_pkg.q_g_s),
        E_l_s=float(iface_pkg.E_l_s),
        E_g_s=float(iface_pkg.E_g_s),
        q_l_cond=float(iface_pkg.q_l_cond),
        q_g_cond=float(iface_pkg.q_g_cond),
        q_l_species_diff=float(iface_pkg.q_l_species_diff),
        q_g_species_diff=float(iface_pkg.q_g_species_diff),
        diagnostics=diagnostics,
    )
