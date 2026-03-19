from __future__ import annotations

"""Generalized Fourier-law energy flux combiner.

Formal paper_v1 contract:
- The only heat-flux law implemented here is q = -k * dT/dr + sum_i(J_i * h_i).
- This module must not compute J_diff_full or grad_T on its own.
- This module must not add latent heat, radiation, or convective enthalpy terms.
- The species-diffusion enthalpy term must never be dropped.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np


class EnergyFluxError(ValueError):
    """Raised when energy-flux inputs are structurally inconsistent."""


@dataclass(frozen=True)
class EnergyFluxPackage:
    face_indices: np.ndarray
    k_face: np.ndarray
    grad_T_face: np.ndarray
    J_diff_full: np.ndarray
    h_face_full: np.ndarray
    q_cond: np.ndarray
    q_species_diff: np.ndarray
    q_total: np.ndarray
    area_face: np.ndarray | None
    Q_cond: np.ndarray | None
    Q_species_diff: np.ndarray | None
    Q_total: np.ndarray | None


def _as_1d_float_array(name: str, value: Any, *, expected_size: int | None = None) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim != 1:
        raise EnergyFluxError(f"{name} must be a 1D float array")
    if not np.all(np.isfinite(arr)):
        raise EnergyFluxError(f"{name} must contain only finite values")
    if expected_size is not None and arr.shape[0] != expected_size:
        raise EnergyFluxError(f"{name} must have length {expected_size}")
    return arr


def _as_1d_int_array(name: str, value: Any) -> np.ndarray:
    arr = np.asarray(value)
    if arr.ndim != 1:
        raise EnergyFluxError(f"{name} must be a 1D integer array")
    if not np.issubdtype(arr.dtype, np.integer):
        raise EnergyFluxError(f"{name} must contain integers")
    return np.asarray(arr, dtype=np.int64)


def _as_2d_float_array(
    name: str,
    value: Any,
    *,
    expected_rows: int | None = None,
    expected_cols: int | None = None,
) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim != 2:
        raise EnergyFluxError(f"{name} must be a 2D float array")
    if not np.all(np.isfinite(arr)):
        raise EnergyFluxError(f"{name} must contain only finite values")
    if expected_rows is not None and arr.shape[0] != expected_rows:
        raise EnergyFluxError(f"{name} must have row count {expected_rows}")
    if expected_cols is not None and arr.shape[1] != expected_cols:
        raise EnergyFluxError(f"{name} must have column count {expected_cols}")
    return arr


def _validate_energy_flux_inputs(
    face_indices: Any,
    k_face: Any,
    grad_T_face: Any,
    J_diff_full: Any,
    h_face_full: Any,
    area_face: Any | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    face_idx = _as_1d_int_array("face_indices", face_indices)
    n_face = face_idx.shape[0]

    k = _as_1d_float_array("k_face", k_face, expected_size=n_face)
    if np.any(k < 0.0):
        raise EnergyFluxError("k_face must be non-negative")

    grad_T = _as_1d_float_array("grad_T_face", grad_T_face, expected_size=n_face)
    J = _as_2d_float_array("J_diff_full", J_diff_full, expected_rows=n_face)
    h = _as_2d_float_array("h_face_full", h_face_full, expected_rows=n_face, expected_cols=J.shape[1])

    area = None
    if area_face is not None:
        area = _as_1d_float_array("area_face", area_face, expected_size=n_face)
        if np.any(area < 0.0):
            raise EnergyFluxError("area_face must be non-negative")

    return face_idx, k, grad_T, J, h, area


def _compute_conductive_heat_flux(
    k_face: np.ndarray,
    grad_T_face: np.ndarray,
) -> np.ndarray:
    return -k_face * grad_T_face


def _compute_species_diffusion_enthalpy_flux(
    J_diff_full: np.ndarray,
    h_face_full: np.ndarray,
) -> np.ndarray:
    return np.sum(J_diff_full * h_face_full, axis=1)


def _compute_total_heat_flux(
    q_cond: np.ndarray,
    q_species_diff: np.ndarray,
) -> np.ndarray:
    return q_cond + q_species_diff


def _multiply_by_face_area(
    q_face: np.ndarray,
    area_face: np.ndarray | None,
) -> np.ndarray | None:
    if area_face is None:
        return None
    return q_face * area_face


def compute_total_energy_flux_density(
    k_face: Any,
    grad_T_face: Any,
    J_diff_full: Any,
    h_face_full: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    k = _as_1d_float_array("k_face", k_face)
    grad_T = _as_1d_float_array("grad_T_face", grad_T_face, expected_size=k.shape[0])
    if np.any(k < 0.0):
        raise EnergyFluxError("k_face must be non-negative")
    J = _as_2d_float_array("J_diff_full", J_diff_full, expected_rows=k.shape[0])
    h = _as_2d_float_array("h_face_full", h_face_full, expected_rows=k.shape[0], expected_cols=J.shape[1])

    q_cond = _compute_conductive_heat_flux(k, grad_T)
    q_species_diff = _compute_species_diffusion_enthalpy_flux(J, h)
    q_total = _compute_total_heat_flux(q_cond, q_species_diff)
    return q_cond, q_species_diff, q_total


def build_energy_flux_package(
    face_indices: Any,
    k_face: Any,
    grad_T_face: Any,
    J_diff_full: Any,
    h_face_full: Any,
    area_face: Any | None = None,
) -> EnergyFluxPackage:
    face_idx, k, grad_T, J, h, area = _validate_energy_flux_inputs(
        face_indices=face_indices,
        k_face=k_face,
        grad_T_face=grad_T_face,
        J_diff_full=J_diff_full,
        h_face_full=h_face_full,
        area_face=area_face,
    )
    q_cond, q_species_diff, q_total = compute_total_energy_flux_density(
        k_face=k,
        grad_T_face=grad_T,
        J_diff_full=J,
        h_face_full=h,
    )
    Q_cond = _multiply_by_face_area(q_cond, area)
    Q_species_diff = _multiply_by_face_area(q_species_diff, area)
    Q_total = _multiply_by_face_area(q_total, area)
    return EnergyFluxPackage(
        face_indices=face_idx,
        k_face=k,
        grad_T_face=grad_T,
        J_diff_full=J,
        h_face_full=h,
        q_cond=q_cond,
        q_species_diff=q_species_diff,
        q_total=q_total,
        area_face=area,
        Q_cond=Q_cond,
        Q_species_diff=Q_species_diff,
        Q_total=Q_total,
    )


__all__ = [
    "EnergyFluxError",
    "EnergyFluxPackage",
    "build_energy_flux_package",
    "compute_total_energy_flux_density",
]
