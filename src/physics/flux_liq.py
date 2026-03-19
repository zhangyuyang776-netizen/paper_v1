from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from core.types import Mesh1D, State


class LiquidFluxError(ValueError):
    """Raised when liquid internal-face fluxes cannot be constructed consistently."""


@dataclass(frozen=True)
class LiquidFaceDiffusionPackage:
    left_cells: np.ndarray
    right_cells: np.ndarray
    face_indices: np.ndarray
    grad_Y_full: np.ndarray
    J_diff_full: np.ndarray
    rho_face: np.ndarray
    D_face_full: np.ndarray


def _validate_liquid_closure_index(*, n_species: int, liquid_closure_index: int | None) -> int | None:
    if n_species == 1:
        return None
    if liquid_closure_index is None:
        raise LiquidFluxError("multicomponent liquid diffusion requires an explicit liquid_closure_index")
    if not isinstance(liquid_closure_index, int):
        raise LiquidFluxError("liquid_closure_index must be an int for multicomponent liquid")
    if liquid_closure_index < 0 or liquid_closure_index >= n_species:
        raise LiquidFluxError("liquid_closure_index must be a valid liquid species index")
    return liquid_closure_index


@dataclass(frozen=True)
class LiquidFaceEnergyPackage:
    left_cells: np.ndarray
    right_cells: np.ndarray
    face_indices: np.ndarray
    grad_T: np.ndarray
    q_cond: np.ndarray
    h_face_full: np.ndarray
    q_species_diff: np.ndarray
    q_total: np.ndarray
    k_face: np.ndarray


def _as_1d_float_array(name: str, value: Any, *, expected_size: int | None = None) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim != 1:
        raise LiquidFluxError(f"{name} must be a 1D float array")
    if not np.all(np.isfinite(arr)):
        raise LiquidFluxError(f"{name} must contain only finite values")
    if expected_size is not None and arr.shape[0] != expected_size:
        raise LiquidFluxError(f"{name} must have length {expected_size}")
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
        raise LiquidFluxError(f"{name} must be a 2D float array")
    if not np.all(np.isfinite(arr)):
        raise LiquidFluxError(f"{name} must contain only finite values")
    if expected_rows is not None and arr.shape[0] != expected_rows:
        raise LiquidFluxError(f"{name} must have row count {expected_rows}")
    if expected_cols is not None and arr.shape[1] != expected_cols:
        raise LiquidFluxError(f"{name} must have column count {expected_cols}")
    return arr


def _validate_state_mesh_compatibility(mesh: Mesh1D, state: State) -> None:
    if state.n_liq_cells != mesh.n_liq:
        raise LiquidFluxError("state liquid cell count must match mesh.n_liq")
    if state.n_gas_cells != mesh.n_gas:
        raise LiquidFluxError("state gas cell count must match mesh.n_gas")
    if state.n_liq_species_full != state.interface.Ys_l_full.shape[0]:
        raise LiquidFluxError("state liquid species dimension must match interface liquid species length")


def _liquid_internal_face_connectivity(mesh: Mesh1D) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_liq = mesh.n_liq
    if n_liq < 1:
        raise LiquidFluxError("mesh.n_liq must be >= 1")
    if mesh.interface_face_index != n_liq:
        raise LiquidFluxError("mesh.interface_face_index must equal mesh.n_liq for liquid face indexing")
    if n_liq == 1:
        empty = np.zeros(0, dtype=np.int64)
        return empty, empty, empty
    left_cells = np.arange(0, n_liq - 1, dtype=np.int64)
    right_cells = np.arange(1, n_liq, dtype=np.int64)
    face_indices = np.arange(1, n_liq, dtype=np.int64)
    return left_cells, right_cells, face_indices


def _central_gradient_scalar(phi: np.ndarray, r_centers: np.ndarray, left: np.ndarray, right: np.ndarray) -> np.ndarray:
    phi_arr = _as_1d_float_array("phi", phi)
    r_arr = _as_1d_float_array("r_centers", r_centers, expected_size=phi_arr.shape[0])
    if left.shape != right.shape:
        raise LiquidFluxError("left/right connectivity arrays must have identical shape")
    if left.size == 0:
        return np.zeros(0, dtype=np.float64)
    dr = r_arr[right] - r_arr[left]
    if np.any(dr <= 0.0):
        raise LiquidFluxError("liquid internal center spacing must be strictly positive")
    return (phi_arr[right] - phi_arr[left]) / dr


def _central_gradient_vector(phi: np.ndarray, r_centers: np.ndarray, left: np.ndarray, right: np.ndarray) -> np.ndarray:
    phi_arr = _as_2d_float_array("phi", phi)
    r_arr = _as_1d_float_array("r_centers", r_centers, expected_size=phi_arr.shape[0])
    if left.shape != right.shape:
        raise LiquidFluxError("left/right connectivity arrays must have identical shape")
    if left.size == 0:
        return np.zeros((0, phi_arr.shape[1]), dtype=np.float64)
    dr = (r_arr[right] - r_arr[left])[:, None]
    if np.any(dr <= 0.0):
        raise LiquidFluxError("liquid internal center spacing must be strictly positive")
    return (phi_arr[right, :] - phi_arr[left, :]) / dr


def _avg_face_scalar(phi: np.ndarray, left: np.ndarray, right: np.ndarray) -> np.ndarray:
    phi_arr = _as_1d_float_array("phi", phi)
    if left.shape != right.shape:
        raise LiquidFluxError("left/right connectivity arrays must have identical shape")
    if left.size == 0:
        return np.zeros(0, dtype=np.float64)
    return 0.5 * (phi_arr[left] + phi_arr[right])


def _avg_face_vector(phi: np.ndarray, left: np.ndarray, right: np.ndarray) -> np.ndarray:
    phi_arr = _as_2d_float_array("phi", phi)
    if left.shape != right.shape:
        raise LiquidFluxError("left/right connectivity arrays must have identical shape")
    if left.size == 0:
        return np.zeros((0, phi_arr.shape[1]), dtype=np.float64)
    return 0.5 * (phi_arr[left, :] + phi_arr[right, :])


def _liquid_cell_density(mesh: Mesh1D, state: State, liquid_props: Any) -> np.ndarray:
    if state.rho_l is not None:
        rho_l = _as_1d_float_array("state.rho_l", state.rho_l, expected_size=mesh.n_liq)
        if np.any(rho_l <= 0.0):
            raise LiquidFluxError("state.rho_l must be strictly positive")
        return rho_l
    rho_l = np.asarray(liquid_props.density_mass_batch(state.Tl, state.Yl_full), dtype=np.float64)
    if rho_l.ndim != 1 or rho_l.shape[0] != mesh.n_liq:
        raise LiquidFluxError("liquid_props.density_mass_batch returned inconsistent shape")
    if not np.all(np.isfinite(rho_l)) or np.any(rho_l <= 0.0):
        raise LiquidFluxError("liquid density field must be finite and strictly positive")
    return rho_l


def _liquid_cell_conductivity(mesh: Mesh1D, state: State, liquid_props: Any) -> np.ndarray:
    values = np.array(
        [liquid_props.conductivity(float(state.Tl[i]), state.Yl_full[i, :]) for i in range(mesh.n_liq)],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
        raise LiquidFluxError("liquid conductivity field must be finite and strictly positive")
    return values


def _liquid_cell_species_enthalpy(mesh: Mesh1D, state: State, liquid_props: Any) -> np.ndarray:
    # Contract with properties/liquid.py:
    # pure_enthalpy_vector(T) must return a full-order pure-species enthalpy vector,
    # mass-based in J/kg, aligned with state.Yl_full species ordering.
    values = np.array(
        [liquid_props.pure_enthalpy_vector(float(state.Tl[i])) for i in range(mesh.n_liq)],
        dtype=np.float64,
    )
    return _as_2d_float_array(
        "liquid species enthalpy field",
        values,
        expected_rows=mesh.n_liq,
        expected_cols=state.n_liq_species_full,
    )


def _liquid_cell_diffusivity(mesh: Mesh1D, state: State, liquid_props: Any) -> np.ndarray:
    n_liq = mesh.n_liq
    n_spec = state.n_liq_species_full
    if n_spec == 1:
        return np.zeros((n_liq, 1), dtype=np.float64)

    rows = []
    for i in range(n_liq):
        diff_i = liquid_props.diffusivity(float(state.Tl[i]), state.Yl_full[i, :])
        if diff_i is None:
            raise LiquidFluxError("liquid_props.diffusivity returned None for a multicomponent liquid state")
        diff_vec = _as_1d_float_array("liquid diffusivity row", diff_i, expected_size=n_spec)
        if np.any(diff_vec <= 0.0):
            raise LiquidFluxError("liquid diffusivity must be strictly positive for multicomponent liquid")
        rows.append(diff_vec)
    return np.vstack(rows).astype(np.float64, copy=False)


def _validate_diffusion_package(mesh: Mesh1D, state: State, diff_pkg: LiquidFaceDiffusionPackage) -> None:
    left_cells, right_cells, face_indices = _liquid_internal_face_connectivity(mesh)
    n_face = face_indices.shape[0]
    n_spec = state.n_liq_species_full

    if not (
        np.array_equal(diff_pkg.left_cells, left_cells)
        and np.array_equal(diff_pkg.right_cells, right_cells)
        and np.array_equal(diff_pkg.face_indices, face_indices)
    ):
        raise LiquidFluxError("diff_pkg connectivity must match mesh liquid internal-face connectivity")

    grad_Y_full = _as_2d_float_array(
        "diff_pkg.grad_Y_full",
        diff_pkg.grad_Y_full,
        expected_rows=n_face,
        expected_cols=n_spec,
    )
    J_diff_full = _as_2d_float_array(
        "diff_pkg.J_diff_full",
        diff_pkg.J_diff_full,
        expected_rows=n_face,
        expected_cols=n_spec,
    )
    D_face_full = _as_2d_float_array(
        "diff_pkg.D_face_full",
        diff_pkg.D_face_full,
        expected_rows=n_face,
        expected_cols=n_spec,
    )
    rho_face = _as_1d_float_array(
        "diff_pkg.rho_face",
        diff_pkg.rho_face,
        expected_size=n_face,
    )
    if np.any(rho_face <= 0.0):
        raise LiquidFluxError("diff_pkg.rho_face must be strictly positive")
    if np.any(D_face_full < 0.0):
        raise LiquidFluxError("diff_pkg.D_face_full must be non-negative")
    _ = grad_Y_full
    _ = J_diff_full


def build_liquid_internal_diffusion_package(
    mesh: Mesh1D,
    state: State,
    liquid_props: Any,
    *,
    liquid_closure_index: int | None = None,
) -> LiquidFaceDiffusionPackage:
    # Formal paper_v1 rule:
    # this module handles liquid internal faces only, not the interface face.
    # Formal paper_v1 rule:
    # liquid diffusivity must be obtained from the properties layer, not recomputed here.
    # Formal paper_v1 rule:
    # multicomponent liquid diffusive fluxes must satisfy sum_i J_i = 0.
    _validate_state_mesh_compatibility(mesh, state)
    left_cells, right_cells, face_indices = _liquid_internal_face_connectivity(mesh)
    r_liq = mesh.r_centers[mesh.liq_slice]
    closure_index = _validate_liquid_closure_index(
        n_species=state.n_liq_species_full,
        liquid_closure_index=liquid_closure_index,
    )

    grad_Y_full = _central_gradient_vector(state.Yl_full, r_liq, left_cells, right_cells)
    rho_cell = _liquid_cell_density(mesh, state, liquid_props)
    rho_face = _avg_face_scalar(rho_cell, left_cells, right_cells)
    D_cell_full = _liquid_cell_diffusivity(mesh, state, liquid_props)
    D_face_full = _avg_face_vector(D_cell_full, left_cells, right_cells)
    J_diff_full = -rho_face[:, None] * D_face_full * grad_Y_full
    if closure_index is not None and J_diff_full.shape[0] > 0:
        nonclosure_mask = np.ones(state.n_liq_species_full, dtype=bool)
        nonclosure_mask[closure_index] = False
        J_diff_full[:, closure_index] = -np.sum(J_diff_full[:, nonclosure_mask], axis=1)

    return LiquidFaceDiffusionPackage(
        left_cells=left_cells,
        right_cells=right_cells,
        face_indices=face_indices,
        grad_Y_full=grad_Y_full,
        J_diff_full=J_diff_full,
        rho_face=rho_face,
        D_face_full=D_face_full,
    )


def build_liquid_internal_energy_flux_package(
    mesh: Mesh1D,
    state: State,
    liquid_props: Any,
    diff_pkg: LiquidFaceDiffusionPackage,
) -> LiquidFaceEnergyPackage:
    # Formal paper_v1 rule:
    # this module handles liquid internal faces only, not the interface face.
    # Formal paper_v1 rule:
    # convective liquid fluxes are handled elsewhere; this module is diffusion/thermal only.
    _validate_state_mesh_compatibility(mesh, state)
    _validate_diffusion_package(mesh, state, diff_pkg)
    left_cells = diff_pkg.left_cells
    right_cells = diff_pkg.right_cells
    face_indices = diff_pkg.face_indices

    r_liq = mesh.r_centers[mesh.liq_slice]
    grad_T = _central_gradient_scalar(state.Tl, r_liq, left_cells, right_cells)
    k_cell = _liquid_cell_conductivity(mesh, state, liquid_props)
    k_face = _avg_face_scalar(k_cell, left_cells, right_cells)
    q_cond = -k_face * grad_T

    h_cell_full = _liquid_cell_species_enthalpy(mesh, state, liquid_props)
    h_face_full = _avg_face_vector(h_cell_full, left_cells, right_cells)
    q_species_diff = np.sum(diff_pkg.J_diff_full * h_face_full, axis=1)
    q_total = q_cond + q_species_diff

    return LiquidFaceEnergyPackage(
        left_cells=left_cells,
        right_cells=right_cells,
        face_indices=face_indices,
        grad_T=grad_T,
        q_cond=q_cond,
        h_face_full=h_face_full,
        q_species_diff=q_species_diff,
        q_total=q_total,
        k_face=k_face,
    )


def build_liquid_center_boundary_flux(n_liq_full: int) -> tuple[np.ndarray, float]:
    if not isinstance(n_liq_full, int) or n_liq_full < 1:
        raise LiquidFluxError("n_liq_full must be an integer >= 1")
    return np.zeros(n_liq_full, dtype=np.float64), 0.0


__all__ = [
    "LiquidFluxError",
    "LiquidFaceDiffusionPackage",
    "LiquidFaceEnergyPackage",
    "build_liquid_internal_diffusion_package",
    "build_liquid_internal_energy_flux_package",
    "build_liquid_center_boundary_flux",
]
