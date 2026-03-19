from __future__ import annotations

"""Convective face-flux kernels based on relative control-face velocity.

Formal paper_v1 boundary rules:
- Convective upwinding must use u_rel = u_abs - v_c, not u_abs alone.
- This module must not reconstruct or infer v_c from geometry; callers provide it.
- This module handles ordinary internal faces and special center/far-field faces only.
- The interface face is excluded and must be handled by interface_face.py.
- This module is field-agnostic numerically, but callers must pass the correct
  transported conservative quantity for the target residual.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np

from core.types import Mesh1D


class ConvectiveFluxError(ValueError):
    """Raised when convective face fluxes cannot be built consistently."""


@dataclass(frozen=True)
class ConvectiveInternalFluxPackage:
    left_cells: np.ndarray
    right_cells: np.ndarray
    face_indices: np.ndarray
    area_face: np.ndarray
    u_abs_face: np.ndarray
    vc_face: np.ndarray
    u_rel_face: np.ndarray
    upwind_is_left: np.ndarray
    phi_upwind: np.ndarray
    flux: np.ndarray


@dataclass(frozen=True)
class BoundaryConvectiveFluxPackage:
    face_index: int
    cell_index: int
    area_face: float
    u_abs_face: float
    vc_face: float
    u_rel_face: float
    upwind_from_cell: bool
    phi_face: np.ndarray
    flux: np.ndarray


def _as_1d_float_array(name: str, value: Any, *, expected_size: int | None = None) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim != 1:
        raise ConvectiveFluxError(f"{name} must be a 1D float array")
    if not np.all(np.isfinite(arr)):
        raise ConvectiveFluxError(f"{name} must contain only finite values")
    if expected_size is not None and arr.shape[0] != expected_size:
        raise ConvectiveFluxError(f"{name} must have length {expected_size}")
    return arr


def _as_2d_float_array(
    name: str,
    value: Any,
    *,
    expected_rows: int | None = None,
    expected_cols: int | None = None,
) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim != 2:
        raise ConvectiveFluxError(f"{name} must be a 2D float array")
    if not np.all(np.isfinite(arr)):
        raise ConvectiveFluxError(f"{name} must contain only finite values")
    if expected_rows is not None and arr.shape[0] != expected_rows:
        raise ConvectiveFluxError(f"{name} must have row count {expected_rows}")
    if expected_cols is not None and arr.shape[1] != expected_cols:
        raise ConvectiveFluxError(f"{name} must have column count {expected_cols}")
    return arr


def _as_scalar_or_vector_1d(name: str, value: Any) -> np.ndarray:
    if np.isscalar(value) and not isinstance(value, bool):
        scalar = float(value)
        if not np.isfinite(scalar):
            raise ConvectiveFluxError(f"{name} must be finite")
        return np.array([scalar], dtype=np.float64)
    return _as_1d_float_array(name, value)


def _compute_relative_face_velocity(u_abs_face: np.ndarray, vc_face: np.ndarray) -> np.ndarray:
    u_abs = _as_1d_float_array("u_abs_face", u_abs_face)
    vc = _as_1d_float_array("vc_face", vc_face, expected_size=u_abs.shape[0])
    return u_abs - vc


def _upwind_select_scalar(
    phi_left: np.ndarray,
    phi_right: np.ndarray,
    u_rel_face: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    left = _as_1d_float_array("phi_left", phi_left)
    right = _as_1d_float_array("phi_right", phi_right, expected_size=left.shape[0])
    u_rel = _as_1d_float_array("u_rel_face", u_rel_face, expected_size=left.shape[0])
    upwind_is_left = u_rel >= 0.0
    phi_upwind = np.where(upwind_is_left, left, right)
    return phi_upwind, upwind_is_left


def _upwind_select_vector(
    phi_left: np.ndarray,
    phi_right: np.ndarray,
    u_rel_face: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    left = _as_2d_float_array("phi_left", phi_left)
    right = _as_2d_float_array("phi_right", phi_right, expected_rows=left.shape[0], expected_cols=left.shape[1])
    u_rel = _as_1d_float_array("u_rel_face", u_rel_face, expected_size=left.shape[0])
    upwind_is_left = u_rel >= 0.0
    phi_upwind = np.where(upwind_is_left[:, None], left, right)
    return phi_upwind, upwind_is_left


def _convective_flux_from_upwind_state(
    area_face: np.ndarray,
    u_rel_face: np.ndarray,
    phi_upwind: np.ndarray,
) -> np.ndarray:
    area = _as_1d_float_array("area_face", area_face)
    u_rel = _as_1d_float_array("u_rel_face", u_rel_face, expected_size=area.shape[0])
    phi = np.asarray(phi_upwind, dtype=np.float64)
    if phi.ndim == 1:
        if phi.shape[0] != area.shape[0]:
            raise ConvectiveFluxError("scalar phi_upwind length must match face count")
        if not np.all(np.isfinite(phi)):
            raise ConvectiveFluxError("phi_upwind must be finite")
        return area * u_rel * phi
    if phi.ndim == 2:
        if phi.shape[0] != area.shape[0]:
            raise ConvectiveFluxError("vector phi_upwind row count must match face count")
        if not np.all(np.isfinite(phi)):
            raise ConvectiveFluxError("phi_upwind must be finite")
        return area[:, None] * u_rel[:, None] * phi
    raise ConvectiveFluxError("phi_upwind must be either a 1D or 2D float array")


def _liquid_internal_face_connectivity(mesh: Mesh1D) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_liq = mesh.n_liq
    if n_liq < 1:
        raise ConvectiveFluxError("mesh.n_liq must be >= 1")
    if n_liq == 1:
        empty = np.zeros(0, dtype=np.int64)
        return empty, empty, empty
    left_cells = np.arange(0, n_liq - 1, dtype=np.int64)
    right_cells = np.arange(1, n_liq, dtype=np.int64)
    face_indices = np.arange(1, n_liq, dtype=np.int64)
    return left_cells, right_cells, face_indices


def _gas_internal_face_connectivity(mesh: Mesh1D) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_gas = mesh.n_gas
    if n_gas < 1:
        raise ConvectiveFluxError("mesh.n_gas must be >= 1")
    if n_gas == 1:
        empty = np.zeros(0, dtype=np.int64)
        return empty, empty, empty
    left_cells = np.arange(0, n_gas - 1, dtype=np.int64)
    right_cells = np.arange(1, n_gas, dtype=np.int64)
    face_indices = np.arange(mesh.interface_face_index + 1, mesh.interface_face_index + n_gas, dtype=np.int64)
    return left_cells, right_cells, face_indices


def _build_internal_convective_flux_scalar(
    *,
    phi_cell: np.ndarray,
    u_abs_face_local: np.ndarray,
    vc_face_local: np.ndarray,
    left_cells: np.ndarray,
    right_cells: np.ndarray,
    face_indices: np.ndarray,
    area_face: np.ndarray,
) -> ConvectiveInternalFluxPackage:
    phi = _as_1d_float_array("phi_cell", phi_cell)
    u_abs_local = _as_1d_float_array("u_abs_face_local", u_abs_face_local)
    vc_local = _as_1d_float_array("vc_face_local", vc_face_local, expected_size=u_abs_local.shape[0])
    if left_cells.size == 0:
        empty = np.zeros(0, dtype=np.float64)
        empty_int = np.zeros(0, dtype=np.int64)
        return ConvectiveInternalFluxPackage(
            left_cells=left_cells,
            right_cells=right_cells,
            face_indices=face_indices,
            area_face=empty,
            u_abs_face=empty,
            vc_face=empty,
            u_rel_face=empty,
            upwind_is_left=np.zeros(0, dtype=bool),
            phi_upwind=empty,
            flux=empty,
        )
    if np.max(right_cells) >= phi.shape[0]:
        raise ConvectiveFluxError("cell connectivity exceeds scalar phi_cell length")
    # Phase-local face-velocity arrays use local ordering with:
    # index 0 reserved for the special face and ordinary internal faces at 1..n_internal.
    local_face_indices = np.arange(1, left_cells.shape[0] + 1, dtype=np.int64)
    u_abs_face = u_abs_local[local_face_indices]
    vc_face = vc_local[local_face_indices]
    u_rel_face = _compute_relative_face_velocity(u_abs_face, vc_face)
    phi_upwind, upwind_is_left = _upwind_select_scalar(phi[left_cells], phi[right_cells], u_rel_face)
    flux = _convective_flux_from_upwind_state(area_face, u_rel_face, phi_upwind)
    return ConvectiveInternalFluxPackage(
        left_cells=left_cells,
        right_cells=right_cells,
        face_indices=face_indices,
        area_face=area_face,
        u_abs_face=u_abs_face,
        vc_face=vc_face,
        u_rel_face=u_rel_face,
        upwind_is_left=upwind_is_left,
        phi_upwind=phi_upwind,
        flux=flux,
    )


def _build_internal_convective_flux_vector(
    *,
    phi_cell: np.ndarray,
    u_abs_face_local: np.ndarray,
    vc_face_local: np.ndarray,
    left_cells: np.ndarray,
    right_cells: np.ndarray,
    face_indices: np.ndarray,
    area_face: np.ndarray,
) -> ConvectiveInternalFluxPackage:
    phi = _as_2d_float_array("phi_cell", phi_cell)
    u_abs_local = _as_1d_float_array("u_abs_face_local", u_abs_face_local)
    vc_local = _as_1d_float_array("vc_face_local", vc_face_local, expected_size=u_abs_local.shape[0])
    if left_cells.size == 0:
        empty = np.zeros(0, dtype=np.float64)
        empty_2d = np.zeros((0, phi.shape[1]), dtype=np.float64)
        return ConvectiveInternalFluxPackage(
            left_cells=left_cells,
            right_cells=right_cells,
            face_indices=face_indices,
            area_face=empty,
            u_abs_face=empty,
            vc_face=empty,
            u_rel_face=empty,
            upwind_is_left=np.zeros(0, dtype=bool),
            phi_upwind=empty_2d,
            flux=empty_2d,
        )
    if np.max(right_cells) >= phi.shape[0]:
        raise ConvectiveFluxError("cell connectivity exceeds vector phi_cell row count")
    local_face_indices = np.arange(1, left_cells.shape[0] + 1, dtype=np.int64)
    u_abs_face = u_abs_local[local_face_indices]
    vc_face = vc_local[local_face_indices]
    u_rel_face = _compute_relative_face_velocity(u_abs_face, vc_face)
    phi_upwind, upwind_is_left = _upwind_select_vector(phi[left_cells, :], phi[right_cells, :], u_rel_face)
    flux = _convective_flux_from_upwind_state(area_face, u_rel_face, phi_upwind)
    return ConvectiveInternalFluxPackage(
        left_cells=left_cells,
        right_cells=right_cells,
        face_indices=face_indices,
        area_face=area_face,
        u_abs_face=u_abs_face,
        vc_face=vc_face,
        u_rel_face=u_rel_face,
        upwind_is_left=upwind_is_left,
        phi_upwind=phi_upwind,
        flux=flux,
    )


def build_liquid_internal_convective_flux_scalar(
    mesh: Mesh1D,
    phi_cell: np.ndarray,
    u_abs_face_liq: np.ndarray,
    vc_face_liq: np.ndarray,
) -> ConvectiveInternalFluxPackage:
    left_cells, right_cells, face_indices = _liquid_internal_face_connectivity(mesh)
    area_face = _as_1d_float_array("liquid area_face", mesh.face_areas[face_indices], expected_size=face_indices.shape[0])
    _as_1d_float_array("u_abs_face_liq", u_abs_face_liq, expected_size=mesh.n_liq + 1)
    _as_1d_float_array("vc_face_liq", vc_face_liq, expected_size=mesh.n_liq + 1)
    return _build_internal_convective_flux_scalar(
        phi_cell=phi_cell,
        u_abs_face_local=u_abs_face_liq,
        vc_face_local=vc_face_liq,
        left_cells=left_cells,
        right_cells=right_cells,
        face_indices=face_indices,
        area_face=area_face,
    )


def build_liquid_internal_convective_flux_vector(
    mesh: Mesh1D,
    phi_cell: np.ndarray,
    u_abs_face_liq: np.ndarray,
    vc_face_liq: np.ndarray,
) -> ConvectiveInternalFluxPackage:
    left_cells, right_cells, face_indices = _liquid_internal_face_connectivity(mesh)
    area_face = _as_1d_float_array("liquid area_face", mesh.face_areas[face_indices], expected_size=face_indices.shape[0])
    _as_1d_float_array("u_abs_face_liq", u_abs_face_liq, expected_size=mesh.n_liq + 1)
    _as_1d_float_array("vc_face_liq", vc_face_liq, expected_size=mesh.n_liq + 1)
    return _build_internal_convective_flux_vector(
        phi_cell=phi_cell,
        u_abs_face_local=u_abs_face_liq,
        vc_face_local=vc_face_liq,
        left_cells=left_cells,
        right_cells=right_cells,
        face_indices=face_indices,
        area_face=area_face,
    )


def build_liquid_center_boundary_convective_flux(phi_template: np.ndarray | float) -> BoundaryConvectiveFluxPackage:
    phi_face = _as_scalar_or_vector_1d("phi_template", phi_template)
    flux = np.zeros_like(phi_face)
    return BoundaryConvectiveFluxPackage(
        face_index=0,
        cell_index=0,
        area_face=0.0,
        u_abs_face=0.0,
        vc_face=0.0,
        u_rel_face=0.0,
        upwind_from_cell=True,
        phi_face=phi_face,
        flux=flux,
    )


def build_gas_internal_convective_flux_scalar(
    mesh: Mesh1D,
    phi_cell: np.ndarray,
    u_abs_face_gas: np.ndarray,
    vc_face_gas: np.ndarray,
) -> ConvectiveInternalFluxPackage:
    left_cells, right_cells, face_indices = _gas_internal_face_connectivity(mesh)
    area_face = _as_1d_float_array("gas area_face", mesh.face_areas[face_indices], expected_size=face_indices.shape[0])
    _as_1d_float_array("u_abs_face_gas", u_abs_face_gas, expected_size=mesh.n_gas + 1)
    _as_1d_float_array("vc_face_gas", vc_face_gas, expected_size=mesh.n_gas + 1)
    return _build_internal_convective_flux_scalar(
        phi_cell=phi_cell,
        u_abs_face_local=u_abs_face_gas,
        vc_face_local=vc_face_gas,
        left_cells=left_cells,
        right_cells=right_cells,
        face_indices=face_indices,
        area_face=area_face,
    )


def build_gas_internal_convective_flux_vector(
    mesh: Mesh1D,
    phi_cell: np.ndarray,
    u_abs_face_gas: np.ndarray,
    vc_face_gas: np.ndarray,
) -> ConvectiveInternalFluxPackage:
    left_cells, right_cells, face_indices = _gas_internal_face_connectivity(mesh)
    area_face = _as_1d_float_array("gas area_face", mesh.face_areas[face_indices], expected_size=face_indices.shape[0])
    _as_1d_float_array("u_abs_face_gas", u_abs_face_gas, expected_size=mesh.n_gas + 1)
    _as_1d_float_array("vc_face_gas", vc_face_gas, expected_size=mesh.n_gas + 1)
    return _build_internal_convective_flux_vector(
        phi_cell=phi_cell,
        u_abs_face_local=u_abs_face_gas,
        vc_face_local=vc_face_gas,
        left_cells=left_cells,
        right_cells=right_cells,
        face_indices=face_indices,
        area_face=area_face,
    )


def _build_gas_farfield_boundary_convective_flux(
    *,
    mesh: Mesh1D,
    phi_cell_last: np.ndarray | float,
    phi_bc: np.ndarray | float,
    u_abs_face_far: float,
    vc_face_far: float,
) -> BoundaryConvectiveFluxPackage:
    if mesh.n_gas < 1:
        raise ConvectiveFluxError("mesh.n_gas must be >= 1 for gas far-field convective closure")
    if mesh.n_faces < 1:
        raise ConvectiveFluxError("mesh.n_faces must be >= 1 for gas far-field convective closure")
    phi_cell = _as_scalar_or_vector_1d("phi_cell_last", phi_cell_last)
    phi_bc_arr = _as_scalar_or_vector_1d("phi_bc", phi_bc)
    if phi_bc_arr.shape != phi_cell.shape:
        raise ConvectiveFluxError("phi_bc must have the same shape as phi_cell_last")
    u_abs = float(u_abs_face_far)
    vc = float(vc_face_far)
    if not np.isfinite(u_abs) or not np.isfinite(vc):
        raise ConvectiveFluxError("far-field face velocities must be finite")
    u_rel = u_abs - vc
    upwind_from_cell = u_rel > 0.0
    phi_face = phi_cell if upwind_from_cell else phi_bc_arr
    area_face = float(mesh.face_areas[-1])
    if not np.isfinite(area_face) or area_face < 0.0:
        raise ConvectiveFluxError("gas far-field face area must be finite and non-negative")
    flux = area_face * u_rel * phi_face
    return BoundaryConvectiveFluxPackage(
        face_index=mesh.n_faces - 1,
        cell_index=mesh.n_gas - 1,
        area_face=area_face,
        u_abs_face=u_abs,
        vc_face=vc,
        u_rel_face=u_rel,
        upwind_from_cell=upwind_from_cell,
        phi_face=phi_face.copy(),
        flux=np.asarray(flux, dtype=np.float64).copy(),
    )


def build_gas_farfield_boundary_convective_flux_scalar(
    mesh: Mesh1D,
    phi_cell_last: float,
    phi_bc: float,
    u_abs_face_far: float,
    vc_face_far: float,
) -> BoundaryConvectiveFluxPackage:
    return _build_gas_farfield_boundary_convective_flux(
        mesh=mesh,
        phi_cell_last=phi_cell_last,
        phi_bc=phi_bc,
        u_abs_face_far=u_abs_face_far,
        vc_face_far=vc_face_far,
    )


def build_gas_farfield_boundary_convective_flux_vector(
    mesh: Mesh1D,
    phi_cell_last: np.ndarray,
    phi_bc: np.ndarray,
    u_abs_face_far: float,
    vc_face_far: float,
) -> BoundaryConvectiveFluxPackage:
    return _build_gas_farfield_boundary_convective_flux(
        mesh=mesh,
        phi_cell_last=phi_cell_last,
        phi_bc=phi_bc,
        u_abs_face_far=u_abs_face_far,
        vc_face_far=vc_face_far,
    )


__all__ = [
    "BoundaryConvectiveFluxPackage",
    "ConvectiveFluxError",
    "ConvectiveInternalFluxPackage",
    "build_gas_farfield_boundary_convective_flux_scalar",
    "build_gas_farfield_boundary_convective_flux_vector",
    "build_gas_internal_convective_flux_scalar",
    "build_gas_internal_convective_flux_vector",
    "build_liquid_center_boundary_convective_flux",
    "build_liquid_internal_convective_flux_scalar",
    "build_liquid_internal_convective_flux_vector",
]
