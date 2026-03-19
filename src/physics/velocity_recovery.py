from __future__ import annotations

"""Continuity-based face-velocity recovery on the frozen current geometry.

This module is part of the inner residual evaluation chain. It recovers phase
face mass fluxes and velocities from the discrete continuity recurrence using
current trial densities, old-equivalent cell masses on the current geometry,
the interface mass-flux quantity from InterfaceFacePackage, and the frozen
control-face velocity.

Forbidden here:
- Stefan-type empirical bulk velocity reconstruction
- imposing any independent gas far-field u Dirichlet
- using old-geometry raw masses instead of old-equivalent current-geometry mass
- recomputing iface_pkg.G_g_if_abs
- rebuilding v_c internally
- sending u_abs directly to convection without subtracting v_c
"""

from dataclasses import dataclass
from typing import Any

import numpy as np

from core.types import ConservativeContents, Mesh1D, OldStateOnCurrentGeometry, State
from physics.interface_face import InterfaceFacePackage


class VelocityRecoveryError(ValueError):
    """Raised when velocity recovery inputs are inconsistent."""


@dataclass(frozen=True)
class OldMassOnCurrentGeometry:
    mass_cell_liq: np.ndarray
    mass_cell_gas: np.ndarray


@dataclass(frozen=True)
class PhaseVelocityRecovery:
    G_face_abs: np.ndarray
    rho_face: np.ndarray
    area_face: np.ndarray
    vc_face: np.ndarray
    u_face_abs: np.ndarray
    u_face_rel: np.ndarray
    current_mass_cell: np.ndarray
    old_mass_cell: np.ndarray


@dataclass(frozen=True)
class VelocityRecoveryPackage:
    liquid: PhaseVelocityRecovery
    gas: PhaseVelocityRecovery
    u_l_if_abs: float
    u_g_if_abs: float
    G_g_if_abs: float
    diagnostics: dict[str, Any]


def _as_1d_float_array(name: str, value: Any, *, expected_size: int | None = None) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim != 1:
        raise VelocityRecoveryError(f"{name} must be a 1D float array")
    if not np.all(np.isfinite(arr)):
        raise VelocityRecoveryError(f"{name} must contain only finite values")
    if expected_size is not None and arr.shape[0] != expected_size:
        raise VelocityRecoveryError(f"{name} must have length {expected_size}")
    return arr


def _validate_positive_scalar(name: str, value: float) -> float:
    scalar = float(value)
    if not np.isfinite(scalar):
        raise VelocityRecoveryError(f"{name} must be finite")
    if scalar <= 0.0:
        raise VelocityRecoveryError(f"{name} must be > 0")
    return scalar


def _validate_finite_scalar(name: str, value: float) -> float:
    scalar = float(value)
    if not np.isfinite(scalar):
        raise VelocityRecoveryError(f"{name} must be finite")
    return scalar


def _validate_old_mass_on_current_geometry(
    old_mass_on_current_geometry: Any,
    *,
    n_liq: int,
    n_gas: int,
) -> OldMassOnCurrentGeometry:
    if isinstance(old_mass_on_current_geometry, OldMassOnCurrentGeometry):
        mass_l = _as_1d_float_array(
            "old_mass_on_current_geometry.mass_cell_liq",
            old_mass_on_current_geometry.mass_cell_liq,
            expected_size=n_liq,
        )
        mass_g = _as_1d_float_array(
            "old_mass_on_current_geometry.mass_cell_gas",
            old_mass_on_current_geometry.mass_cell_gas,
            expected_size=n_gas,
        )
        return OldMassOnCurrentGeometry(mass_cell_liq=mass_l.copy(), mass_cell_gas=mass_g.copy())

    if isinstance(old_mass_on_current_geometry, OldStateOnCurrentGeometry):
        contents = old_mass_on_current_geometry.contents
        mass_l = _as_1d_float_array("old_state_current_geom.contents.mass_l", contents.mass_l, expected_size=n_liq)
        mass_g = _as_1d_float_array("old_state_current_geom.contents.mass_g", contents.mass_g, expected_size=n_gas)
        return OldMassOnCurrentGeometry(mass_cell_liq=mass_l.copy(), mass_cell_gas=mass_g.copy())

    if isinstance(old_mass_on_current_geometry, ConservativeContents):
        mass_l = _as_1d_float_array("contents.mass_l", old_mass_on_current_geometry.mass_l, expected_size=n_liq)
        mass_g = _as_1d_float_array("contents.mass_g", old_mass_on_current_geometry.mass_g, expected_size=n_gas)
        return OldMassOnCurrentGeometry(mass_cell_liq=mass_l.copy(), mass_cell_gas=mass_g.copy())

    if hasattr(old_mass_on_current_geometry, "mass_cell_liq") and hasattr(old_mass_on_current_geometry, "mass_cell_gas"):
        mass_l = _as_1d_float_array("old_mass_on_current_geometry.mass_cell_liq", old_mass_on_current_geometry.mass_cell_liq, expected_size=n_liq)
        mass_g = _as_1d_float_array("old_mass_on_current_geometry.mass_cell_gas", old_mass_on_current_geometry.mass_cell_gas, expected_size=n_gas)
        return OldMassOnCurrentGeometry(mass_cell_liq=mass_l.copy(), mass_cell_gas=mass_g.copy())

    raise VelocityRecoveryError(
        "old_mass_on_current_geometry must provide current-geometry old cell masses "
        "via OldMassOnCurrentGeometry, OldStateOnCurrentGeometry, ConservativeContents, "
        "or mass_cell_liq/mass_cell_gas attributes"
    )


def _validate_state_densities(mesh: Mesh1D, state: State) -> tuple[np.ndarray, np.ndarray]:
    if state.rho_l is None or state.rho_g is None:
        raise VelocityRecoveryError("state.rho_l and state.rho_g must be provided for velocity recovery")
    rho_l = _as_1d_float_array("state.rho_l", state.rho_l, expected_size=mesh.n_liq)
    rho_g = _as_1d_float_array("state.rho_g", state.rho_g, expected_size=mesh.n_gas)
    if np.any(rho_l <= 0.0):
        raise VelocityRecoveryError("state.rho_l must be strictly positive")
    if np.any(rho_g <= 0.0):
        raise VelocityRecoveryError("state.rho_g must be strictly positive")
    return rho_l, rho_g


def _liquid_face_areas(mesh: Mesh1D) -> np.ndarray:
    if mesh.interface_face_index is None or mesh.interface_face_index != mesh.n_liq:
        raise VelocityRecoveryError("mesh.interface_face_index must equal mesh.n_liq for local liquid face ordering")
    return _as_1d_float_array("liquid face areas", mesh.face_areas[: mesh.n_liq + 1], expected_size=mesh.n_liq + 1)


def _gas_face_areas(mesh: Mesh1D) -> np.ndarray:
    if mesh.interface_face_index is None or mesh.interface_face_index != mesh.n_liq:
        raise VelocityRecoveryError("mesh.interface_face_index must equal mesh.n_liq for local gas face ordering")
    return _as_1d_float_array("gas face areas", mesh.face_areas[mesh.interface_face_index :], expected_size=mesh.n_gas + 1)


def _liquid_cell_volumes(mesh: Mesh1D) -> np.ndarray:
    return _as_1d_float_array("liquid cell volumes", mesh.volumes[mesh.liq_slice], expected_size=mesh.n_liq)


def _gas_cell_volumes(mesh: Mesh1D) -> np.ndarray:
    return _as_1d_float_array("gas cell volumes", mesh.volumes[mesh.gas_slice], expected_size=mesh.n_gas)


def _build_liquid_face_density(
    rho_cell_liq: np.ndarray,
) -> np.ndarray:
    n_liq = rho_cell_liq.shape[0]
    rho_face = np.empty(n_liq + 1, dtype=np.float64)
    rho_face[0] = rho_cell_liq[0]
    if n_liq > 1:
        rho_face[1:n_liq] = 0.5 * (rho_cell_liq[:-1] + rho_cell_liq[1:])
    rho_face[n_liq] = rho_cell_liq[-1]
    return rho_face


def _build_gas_face_density(
    rho_cell_gas: np.ndarray,
) -> np.ndarray:
    n_gas = rho_cell_gas.shape[0]
    rho_face = np.empty(n_gas + 1, dtype=np.float64)
    rho_face[0] = rho_cell_gas[0]
    if n_gas > 1:
        rho_face[1:n_gas] = 0.5 * (rho_cell_gas[:-1] + rho_cell_gas[1:])
    rho_face[n_gas] = rho_cell_gas[-1]
    return rho_face


def _recover_face_velocity(
    *,
    G_face_abs: np.ndarray,
    rho_face: np.ndarray,
    area_face: np.ndarray,
    vc_face: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    u_face_abs = np.empty_like(G_face_abs)
    for idx in range(G_face_abs.shape[0]):
        denom = rho_face[idx] * area_face[idx]
        if area_face[idx] == 0.0:
            if not np.isclose(G_face_abs[idx], 0.0, rtol=0.0, atol=1.0e-14):
                raise VelocityRecoveryError("zero-area face must carry zero absolute mass flux")
            u_face_abs[idx] = 0.0
            continue
        if denom <= 0.0 or not np.isfinite(denom):
            raise VelocityRecoveryError("rho_face * area_face must be finite and strictly positive on nonzero-area faces")
        u_face_abs[idx] = G_face_abs[idx] / denom
    u_face_rel = u_face_abs - vc_face
    if not np.all(np.isfinite(u_face_rel)):
        raise VelocityRecoveryError("Recovered relative face velocity must be finite")
    return u_face_abs, u_face_rel


def _recover_liquid_phase_velocity(
    rho_cell_liq: np.ndarray,
    old_mass_cell_liq: np.ndarray,
    face_area_liq: np.ndarray,
    cell_volume_liq: np.ndarray,
    vc_face_liq: np.ndarray,
    dt: float,
) -> PhaseVelocityRecovery:
    rho_face = _build_liquid_face_density(rho_cell_liq)
    current_mass_cell = rho_cell_liq * cell_volume_liq
    G_face_abs = np.empty(rho_cell_liq.shape[0] + 1, dtype=np.float64)
    G_face_abs[0] = 0.0

    for n in range(rho_cell_liq.shape[0]):
        f_left = n
        f_right = n + 1
        C_n = rho_face[f_right] * vc_face_liq[f_right] * face_area_liq[f_right] - rho_face[f_left] * vc_face_liq[f_left] * face_area_liq[f_left]
        S_n = (current_mass_cell[n] - old_mass_cell_liq[n]) / dt
        G_face_abs[f_right] = G_face_abs[f_left] + C_n - S_n

    u_face_abs, u_face_rel = _recover_face_velocity(
        G_face_abs=G_face_abs,
        rho_face=rho_face,
        area_face=face_area_liq,
        vc_face=vc_face_liq,
    )
    return PhaseVelocityRecovery(
        G_face_abs=G_face_abs,
        rho_face=rho_face,
        area_face=face_area_liq,
        vc_face=vc_face_liq,
        u_face_abs=u_face_abs,
        u_face_rel=u_face_rel,
        current_mass_cell=current_mass_cell,
        old_mass_cell=old_mass_cell_liq,
    )


def _recover_gas_phase_velocity(
    rho_cell_gas: np.ndarray,
    old_mass_cell_gas: np.ndarray,
    face_area_gas: np.ndarray,
    cell_volume_gas: np.ndarray,
    vc_face_gas: np.ndarray,
    G_g_if_abs: float,
    dt: float,
) -> PhaseVelocityRecovery:
    rho_face = _build_gas_face_density(rho_cell_gas)
    current_mass_cell = rho_cell_gas * cell_volume_gas
    G_face_abs = np.empty(rho_cell_gas.shape[0] + 1, dtype=np.float64)
    G_face_abs[0] = float(G_g_if_abs)

    for n in range(rho_cell_gas.shape[0]):
        f_left = n
        f_right = n + 1
        C_n = rho_face[f_right] * vc_face_gas[f_right] * face_area_gas[f_right] - rho_face[f_left] * vc_face_gas[f_left] * face_area_gas[f_left]
        S_n = (current_mass_cell[n] - old_mass_cell_gas[n]) / dt
        G_face_abs[f_right] = G_face_abs[f_left] + C_n - S_n

    u_face_abs, u_face_rel = _recover_face_velocity(
        G_face_abs=G_face_abs,
        rho_face=rho_face,
        area_face=face_area_gas,
        vc_face=vc_face_gas,
    )
    return PhaseVelocityRecovery(
        G_face_abs=G_face_abs,
        rho_face=rho_face,
        area_face=face_area_gas,
        vc_face=vc_face_gas,
        u_face_abs=u_face_abs,
        u_face_rel=u_face_rel,
        current_mass_cell=current_mass_cell,
        old_mass_cell=old_mass_cell_gas,
    )


def build_velocity_recovery_package(
    mesh: Mesh1D,
    state: State,
    old_mass_on_current_geometry: Any,
    iface_pkg: InterfaceFacePackage,
    vc_face_liq: Any,
    vc_face_gas: Any,
    dt: float,
) -> VelocityRecoveryPackage:
    """Recover phase face mass fluxes and velocities from discrete continuity."""

    dt_value = _validate_positive_scalar("dt", dt)
    rho_l, rho_g = _validate_state_densities(mesh, state)

    old_mass = _validate_old_mass_on_current_geometry(
        old_mass_on_current_geometry,
        n_liq=mesh.n_liq,
        n_gas=mesh.n_gas,
    )

    face_area_liq = _liquid_face_areas(mesh)
    face_area_gas = _gas_face_areas(mesh)
    if np.any(face_area_liq[1:] <= 0.0):
        raise VelocityRecoveryError("non-center liquid face areas must be strictly positive")
    if np.any(face_area_gas <= 0.0):
        raise VelocityRecoveryError("gas face areas must be strictly positive")

    cell_volume_liq = _liquid_cell_volumes(mesh)
    cell_volume_gas = _gas_cell_volumes(mesh)
    if np.any(cell_volume_liq <= 0.0):
        raise VelocityRecoveryError("liquid cell volumes must be strictly positive")
    if np.any(cell_volume_gas <= 0.0):
        raise VelocityRecoveryError("gas cell volumes must be strictly positive")

    vc_l = _as_1d_float_array("vc_face_liq", vc_face_liq, expected_size=mesh.n_liq + 1)
    vc_g = _as_1d_float_array("vc_face_gas", vc_face_gas, expected_size=mesh.n_gas + 1)
    G_g_if_abs = _validate_finite_scalar("iface_pkg.G_g_if_abs", iface_pkg.G_g_if_abs)

    liquid = _recover_liquid_phase_velocity(
        rho_cell_liq=rho_l,
        old_mass_cell_liq=old_mass.mass_cell_liq,
        face_area_liq=face_area_liq,
        cell_volume_liq=cell_volume_liq,
        vc_face_liq=vc_l,
        dt=dt_value,
    )
    gas = _recover_gas_phase_velocity(
        rho_cell_gas=rho_g,
        old_mass_cell_gas=old_mass.mass_cell_gas,
        face_area_gas=face_area_gas,
        cell_volume_gas=cell_volume_gas,
        vc_face_gas=vc_g,
        G_g_if_abs=G_g_if_abs,
        dt=dt_value,
    )

    u_l_if_abs = float(liquid.u_face_abs[-1])
    u_g_if_abs = float(gas.u_face_abs[0])
    diagnostics = {
        "liquid_center_G": float(liquid.G_face_abs[0]),
        "gas_interface_G": float(gas.G_face_abs[0]),
        "u_l_if_abs": u_l_if_abs,
        "u_g_if_abs": u_g_if_abs,
        "u_l_if_rel": float(liquid.u_face_rel[-1]),
        "u_g_if_rel": float(gas.u_face_rel[0]),
    }

    return VelocityRecoveryPackage(
        liquid=liquid,
        gas=gas,
        u_l_if_abs=u_l_if_abs,
        u_g_if_abs=u_g_if_abs,
        G_g_if_abs=G_g_if_abs,
        diagnostics=diagnostics,
    )
