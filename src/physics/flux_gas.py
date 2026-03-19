from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from core.types import Mesh1D, State


class GasFluxError(ValueError):
    """Raised when gas diffusive / thermal face fluxes cannot be built consistently."""


@dataclass(frozen=True)
class GasFaceDiffusionPackage:
    left_cells: np.ndarray
    right_cells: np.ndarray
    face_indices: np.ndarray
    X_face_full: np.ndarray
    Y_face_full: np.ndarray
    dXdr_full: np.ndarray
    rho_face: np.ndarray
    D_face_full: np.ndarray
    Vd0_face_full: np.ndarray
    Vcd_face: np.ndarray
    J_diff_full: np.ndarray


@dataclass(frozen=True)
class GasFaceEnergyPackage:
    left_cells: np.ndarray
    right_cells: np.ndarray
    face_indices: np.ndarray
    grad_T: np.ndarray
    k_face: np.ndarray
    h_face_full: np.ndarray
    q_cond: np.ndarray
    q_species_diff: np.ndarray
    q_total: np.ndarray


@dataclass(frozen=True)
class GasFarfieldFluxPackage:
    face_index: int
    cell_index: int
    X_face_full: np.ndarray
    Y_face_full: np.ndarray
    dXdr_full: np.ndarray
    rho_face: float
    D_face_full: np.ndarray
    Vd0_face_full: np.ndarray
    Vcd_face: float
    J_diff_full: np.ndarray
    grad_T: float
    k_face: float
    h_face_full: np.ndarray
    q_cond: float
    q_species_diff: float
    q_total: float


def _as_1d_float_array(name: str, value: Any, *, expected_size: int | None = None) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim != 1:
        raise GasFluxError(f"{name} must be a 1D float array")
    if not np.all(np.isfinite(arr)):
        raise GasFluxError(f"{name} must contain only finite values")
    if expected_size is not None and arr.shape[0] != expected_size:
        raise GasFluxError(f"{name} must have length {expected_size}")
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
        raise GasFluxError(f"{name} must be a 2D float array")
    if not np.all(np.isfinite(arr)):
        raise GasFluxError(f"{name} must contain only finite values")
    if expected_rows is not None and arr.shape[0] != expected_rows:
        raise GasFluxError(f"{name} must have row count {expected_rows}")
    if expected_cols is not None and arr.shape[1] != expected_cols:
        raise GasFluxError(f"{name} must have column count {expected_cols}")
    return arr


def _validate_state_mesh_compatibility(mesh: Mesh1D, state: State) -> None:
    if state.n_gas_cells != mesh.n_gas:
        raise GasFluxError("state gas cell count must match mesh.n_gas")
    if state.n_liq_cells != mesh.n_liq:
        raise GasFluxError("state liquid cell count must match mesh.n_liq")
    if state.n_gas_species_full != state.interface.Ys_g_full.shape[0]:
        raise GasFluxError("state gas species dimension must match interface gas species length")


def _validate_pressure(name: str, value: float) -> float:
    scalar = float(value)
    if not np.isfinite(scalar) or scalar <= 0.0:
        raise GasFluxError(f"{name} must be finite and > 0")
    return scalar


def _validate_mass_fraction_rows(name: str, arr: np.ndarray, *, atol: float = 1.0e-12) -> np.ndarray:
    values = _as_2d_float_array(name, arr)
    if np.any(values < -atol):
        raise GasFluxError(f"{name} must be non-negative")
    if not np.allclose(np.sum(values, axis=1), 1.0, rtol=0.0, atol=atol):
        raise GasFluxError(f"{name} must sum to 1 along the species axis")
    if np.any(values < 0.0):
        raise GasFluxError(f"{name} contains entries below zero outside numerical tolerance")
    return values


def _mass_to_mole_fractions(Y_full: np.ndarray, molecular_weights: np.ndarray) -> np.ndarray:
    Y = _validate_mass_fraction_rows("Y_full", Y_full)
    mw = _as_1d_float_array("molecular_weights", molecular_weights, expected_size=Y.shape[1])
    if np.any(mw <= 0.0):
        raise GasFluxError("molecular_weights must be strictly positive")
    mole_basis = Y / mw[None, :]
    denom = np.sum(mole_basis, axis=1, keepdims=True)
    if np.any(denom <= 0.0) or not np.all(np.isfinite(denom)):
        raise GasFluxError("mole-fraction denominator must be finite and strictly positive")
    X = mole_basis / denom
    if not np.allclose(np.sum(X, axis=1), 1.0, rtol=0.0, atol=1.0e-12):
        raise GasFluxError("computed gas mole fractions must sum to 1")
    return X


def _gas_internal_face_connectivity(mesh: Mesh1D) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_gas = mesh.n_gas
    if n_gas < 1:
        raise GasFluxError("mesh.n_gas must be >= 1")
    if mesh.interface_face_index != mesh.n_liq:
        raise GasFluxError("mesh.interface_face_index must equal mesh.n_liq")
    if n_gas == 1:
        empty = np.zeros(0, dtype=np.int64)
        return empty, empty, empty
    left_cells = np.arange(0, n_gas - 1, dtype=np.int64)
    right_cells = np.arange(1, n_gas, dtype=np.int64)
    face_indices = np.arange(mesh.interface_face_index + 1, mesh.interface_face_index + n_gas, dtype=np.int64)
    return left_cells, right_cells, face_indices


def _central_gradient_scalar(phi: np.ndarray, r_centers: np.ndarray, left: np.ndarray, right: np.ndarray) -> np.ndarray:
    phi_arr = _as_1d_float_array("phi", phi)
    r_arr = _as_1d_float_array("r_centers", r_centers, expected_size=phi_arr.shape[0])
    if left.size == 0:
        return np.zeros(0, dtype=np.float64)
    dr = r_arr[right] - r_arr[left]
    if np.any(dr <= 0.0):
        raise GasFluxError("gas center spacing must be strictly positive")
    return (phi_arr[right] - phi_arr[left]) / dr


def _central_gradient_vector(phi: np.ndarray, r_centers: np.ndarray, left: np.ndarray, right: np.ndarray) -> np.ndarray:
    phi_arr = _as_2d_float_array("phi", phi)
    r_arr = _as_1d_float_array("r_centers", r_centers, expected_size=phi_arr.shape[0])
    if left.size == 0:
        return np.zeros((0, phi_arr.shape[1]), dtype=np.float64)
    dr = (r_arr[right] - r_arr[left])[:, None]
    if np.any(dr <= 0.0):
        raise GasFluxError("gas center spacing must be strictly positive")
    return (phi_arr[right, :] - phi_arr[left, :]) / dr


def _avg_face_scalar(phi: np.ndarray, left: np.ndarray, right: np.ndarray) -> np.ndarray:
    phi_arr = _as_1d_float_array("phi", phi)
    if left.size == 0:
        return np.zeros(0, dtype=np.float64)
    return 0.5 * (phi_arr[left] + phi_arr[right])


def _avg_face_vector(phi: np.ndarray, left: np.ndarray, right: np.ndarray) -> np.ndarray:
    phi_arr = _as_2d_float_array("phi", phi)
    if left.size == 0:
        return np.zeros((0, phi_arr.shape[1]), dtype=np.float64)
    return 0.5 * (phi_arr[left, :] + phi_arr[right, :])


def _gas_cell_mole_fractions(state: State, gas_props: Any) -> np.ndarray:
    # Phase-2 / paper_v1 contract: gas_props.molecular_weights is a stable
    # public full-order field in kg/mol that physics-side transport kernels may
    # use for Y -> X conversion when a dedicated helper is not provided.
    return _mass_to_mole_fractions(state.Yg_full, np.asarray(gas_props.molecular_weights, dtype=np.float64))


def _gas_cell_density(mesh: Mesh1D, state: State, gas_props: Any, p_env: float) -> np.ndarray:
    if state.rho_g is not None:
        rho_g = _as_1d_float_array("state.rho_g", state.rho_g, expected_size=mesh.n_gas)
        if np.any(rho_g <= 0.0):
            raise GasFluxError("state.rho_g must be strictly positive")
        return rho_g
    rho_g = np.asarray(gas_props.density_mass_batch(state.Tg, state.Yg_full, p_env), dtype=np.float64)
    if rho_g.ndim != 1 or rho_g.shape[0] != mesh.n_gas:
        raise GasFluxError("gas_props.density_mass_batch returned inconsistent shape")
    if not np.all(np.isfinite(rho_g)) or np.any(rho_g <= 0.0):
        raise GasFluxError("gas density field must be finite and strictly positive")
    return rho_g


def _gas_cell_conductivity(mesh: Mesh1D, state: State, gas_props: Any, p_env: float) -> np.ndarray:
    values = np.array(
        [gas_props.conductivity(float(state.Tg[i]), state.Yg_full[i, :], p_env) for i in range(mesh.n_gas)],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
        raise GasFluxError("gas conductivity field must be finite and strictly positive")
    return values


def _gas_cell_diffusivity(mesh: Mesh1D, state: State, gas_props: Any, p_env: float) -> np.ndarray:
    rows = [
        _as_1d_float_array(
            "gas diffusivity row",
            gas_props.diffusivity(float(state.Tg[i]), state.Yg_full[i, :], p_env),
            expected_size=state.n_gas_species_full,
        )
        for i in range(mesh.n_gas)
    ]
    D = np.vstack(rows).astype(np.float64, copy=False)
    if np.any(D <= 0.0):
        raise GasFluxError("gas diffusivity must be strictly positive")
    return D


def _gas_cell_species_enthalpy(mesh: Mesh1D, state: State, gas_props: Any) -> np.ndarray:
    values = np.array([gas_props.species_enthalpies_mass(float(state.Tg[i])) for i in range(mesh.n_gas)], dtype=np.float64)
    return _as_2d_float_array(
        "gas species enthalpy field",
        values,
        expected_rows=mesh.n_gas,
        expected_cols=state.n_gas_species_full,
    )


def _compute_mixture_averaged_diffusion_flux(
    *,
    rho_face: np.ndarray,
    Y_face_full: np.ndarray,
    X_face_full: np.ndarray,
    dXdr_full: np.ndarray,
    D_face_full: np.ndarray,
    x_eps: float = 1.0e-14,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_safe = np.maximum(X_face_full, x_eps)
    Vd0_face_full = -(D_face_full / x_safe) * dXdr_full
    Vcd_face = -np.sum(Y_face_full * Vd0_face_full, axis=1)
    J_diff_full = -rho_face[:, None] * Y_face_full * (Vd0_face_full + Vcd_face[:, None])
    return Vd0_face_full, Vcd_face, J_diff_full


def _validate_internal_diffusion_package(mesh: Mesh1D, state: State, diff_pkg: GasFaceDiffusionPackage) -> None:
    left_cells, right_cells, face_indices = _gas_internal_face_connectivity(mesh)
    n_face = face_indices.shape[0]
    n_spec = state.n_gas_species_full

    if not (
        np.array_equal(diff_pkg.left_cells, left_cells)
        and np.array_equal(diff_pkg.right_cells, right_cells)
        and np.array_equal(diff_pkg.face_indices, face_indices)
    ):
        raise GasFluxError("diff_pkg connectivity must match mesh gas internal-face connectivity")

    _as_2d_float_array("diff_pkg.X_face_full", diff_pkg.X_face_full, expected_rows=n_face, expected_cols=n_spec)
    _as_2d_float_array("diff_pkg.Y_face_full", diff_pkg.Y_face_full, expected_rows=n_face, expected_cols=n_spec)
    _as_2d_float_array("diff_pkg.dXdr_full", diff_pkg.dXdr_full, expected_rows=n_face, expected_cols=n_spec)
    D_face_full = _as_2d_float_array("diff_pkg.D_face_full", diff_pkg.D_face_full, expected_rows=n_face, expected_cols=n_spec)
    J_diff_full = _as_2d_float_array("diff_pkg.J_diff_full", diff_pkg.J_diff_full, expected_rows=n_face, expected_cols=n_spec)
    rho_face = _as_1d_float_array("diff_pkg.rho_face", diff_pkg.rho_face, expected_size=n_face)
    _as_2d_float_array("diff_pkg.Vd0_face_full", diff_pkg.Vd0_face_full, expected_rows=n_face, expected_cols=n_spec)
    _as_1d_float_array("diff_pkg.Vcd_face", diff_pkg.Vcd_face, expected_size=n_face)
    if np.any(rho_face <= 0.0):
        raise GasFluxError("diff_pkg.rho_face must be strictly positive")
    if np.any(D_face_full <= 0.0):
        raise GasFluxError("diff_pkg.D_face_full must be strictly positive")
    if not np.allclose(np.sum(J_diff_full, axis=1), 0.0, rtol=0.0, atol=1.0e-10):
        raise GasFluxError("gas diffusive flux package must satisfy sum_i J_i = 0 on every internal face")


def build_gas_internal_diffusion_package(
    mesh: Mesh1D,
    state: State,
    gas_props: Any,
    p_env: float,
) -> GasFaceDiffusionPackage:
    # Formal paper_v1 rule:
    # gas diffusion uses the mixture-averaged form with conservative correction, not a Fick mass-fraction law.
    # Formal paper_v1 rule:
    # dX/dr must be computed from full-order gas mole fractions, not reduced species or mass-fraction gradients.
    _validate_state_mesh_compatibility(mesh, state)
    p_value = _validate_pressure("p_env", p_env)
    left_cells, right_cells, face_indices = _gas_internal_face_connectivity(mesh)
    r_gas = mesh.r_centers[mesh.gas_slice]

    X_cell_full = _gas_cell_mole_fractions(state, gas_props)
    Y_face_full = _avg_face_vector(state.Yg_full, left_cells, right_cells)
    X_face_full = _avg_face_vector(X_cell_full, left_cells, right_cells)
    dXdr_full = _central_gradient_vector(X_cell_full, r_gas, left_cells, right_cells)
    rho_face = _avg_face_scalar(_gas_cell_density(mesh, state, gas_props, p_value), left_cells, right_cells)
    D_face_full = _avg_face_vector(_gas_cell_diffusivity(mesh, state, gas_props, p_value), left_cells, right_cells)
    Vd0_face_full, Vcd_face, J_diff_full = _compute_mixture_averaged_diffusion_flux(
        rho_face=rho_face,
        Y_face_full=Y_face_full,
        X_face_full=X_face_full,
        dXdr_full=dXdr_full,
        D_face_full=D_face_full,
    )

    pkg = GasFaceDiffusionPackage(
        left_cells=left_cells,
        right_cells=right_cells,
        face_indices=face_indices,
        X_face_full=X_face_full,
        Y_face_full=Y_face_full,
        dXdr_full=dXdr_full,
        rho_face=rho_face,
        D_face_full=D_face_full,
        Vd0_face_full=Vd0_face_full,
        Vcd_face=Vcd_face,
        J_diff_full=J_diff_full,
    )
    _validate_internal_diffusion_package(mesh, state, pkg)
    return pkg


def build_gas_internal_energy_flux_package(
    mesh: Mesh1D,
    state: State,
    gas_props: Any,
    p_env: float,
    diff_pkg: GasFaceDiffusionPackage,
) -> GasFaceEnergyPackage:
    # Formal paper_v1 rule:
    # this module handles gas ordinary internal faces only, not the interface face or convective flux.
    _validate_state_mesh_compatibility(mesh, state)
    _validate_pressure("p_env", p_env)
    _validate_internal_diffusion_package(mesh, state, diff_pkg)
    left_cells = diff_pkg.left_cells
    right_cells = diff_pkg.right_cells
    face_indices = diff_pkg.face_indices
    r_gas = mesh.r_centers[mesh.gas_slice]

    grad_T = _central_gradient_scalar(state.Tg, r_gas, left_cells, right_cells)
    k_face = _avg_face_scalar(_gas_cell_conductivity(mesh, state, gas_props, p_env), left_cells, right_cells)
    q_cond = -k_face * grad_T
    h_face_full = _avg_face_vector(_gas_cell_species_enthalpy(mesh, state, gas_props), left_cells, right_cells)
    q_species_diff = np.sum(diff_pkg.J_diff_full * h_face_full, axis=1)
    q_total = q_cond + q_species_diff

    return GasFaceEnergyPackage(
        left_cells=left_cells,
        right_cells=right_cells,
        face_indices=face_indices,
        grad_T=grad_T,
        k_face=k_face,
        h_face_full=h_face_full,
        q_cond=q_cond,
        q_species_diff=q_species_diff,
        q_total=q_total,
    )


def build_gas_farfield_boundary_flux_package(
    mesh: Mesh1D,
    state: State,
    gas_props: Any,
    p_env: float,
    T_inf: float,
    Yg_inf_full: np.ndarray,
) -> GasFarfieldFluxPackage:
    # Formal paper_v1 rule:
    # far-field gas closure is a special face and must be built separately from ordinary internal faces.
    _validate_state_mesh_compatibility(mesh, state)
    p_value = _validate_pressure("p_env", p_env)
    T_inf_value = float(T_inf)
    if not np.isfinite(T_inf_value) or T_inf_value <= 0.0:
        raise GasFluxError("T_inf must be finite and > 0")
    Y_inf = _as_1d_float_array("Yg_inf_full", Yg_inf_full, expected_size=state.n_gas_species_full)
    if np.any(Y_inf < -1.0e-12) or not np.isclose(float(np.sum(Y_inf)), 1.0, rtol=0.0, atol=1.0e-12):
        raise GasFluxError("Yg_inf_full must be a valid full-order gas mass-fraction vector")

    cell_index = state.n_gas_cells - 1
    face_index = mesh.n_faces - 1
    r_cell = float(mesh.r_centers[mesh.gas_slice][cell_index])
    r_face = float(mesh.r_faces[face_index])
    dr = r_face - r_cell
    if not np.isfinite(dr) or dr <= 0.0:
        raise GasFluxError("far-field one-sided gas spacing must be strictly positive")

    X_cell = _gas_cell_mole_fractions(state, gas_props)[cell_index, :]
    X_inf = _mass_to_mole_fractions(Y_inf[None, :], np.asarray(gas_props.molecular_weights, dtype=np.float64))[0, :]

    Y_face_full = 0.5 * (state.Yg_full[cell_index, :] + Y_inf)
    X_face_full = 0.5 * (X_cell + X_inf)
    dXdr_full = (X_inf - X_cell) / dr
    grad_T = (T_inf_value - float(state.Tg[cell_index])) / dr

    rho_cell = _gas_cell_density(mesh, state, gas_props, p_value)[cell_index]
    rho_inf = float(gas_props.density_mass(T_inf_value, Y_inf, p_value))
    rho_face = 0.5 * (rho_cell + rho_inf)

    D_cell = _gas_cell_diffusivity(mesh, state, gas_props, p_value)[cell_index, :]
    D_inf = _as_1d_float_array(
        "far-field gas diffusivity",
        gas_props.diffusivity(T_inf_value, Y_inf, p_value),
        expected_size=state.n_gas_species_full,
    )
    D_face_full = 0.5 * (D_cell + D_inf)

    Vd0_face_full, Vcd_face_arr, J_diff_full_arr = _compute_mixture_averaged_diffusion_flux(
        rho_face=np.array([rho_face], dtype=np.float64),
        Y_face_full=Y_face_full[None, :],
        X_face_full=X_face_full[None, :],
        dXdr_full=dXdr_full[None, :],
        D_face_full=D_face_full[None, :],
    )
    Vd0_face_full = Vd0_face_full[0, :]
    Vcd_face = float(Vcd_face_arr[0])
    J_diff_full = J_diff_full_arr[0, :]

    k_cell = _gas_cell_conductivity(mesh, state, gas_props, p_value)[cell_index]
    k_inf = float(gas_props.conductivity(T_inf_value, Y_inf, p_value))
    k_face = 0.5 * (k_cell + k_inf)
    q_cond = -k_face * grad_T

    h_cell = _gas_cell_species_enthalpy(mesh, state, gas_props)[cell_index, :]
    h_inf = _as_1d_float_array(
        "far-field gas species enthalpy",
        gas_props.species_enthalpies_mass(T_inf_value),
        expected_size=state.n_gas_species_full,
    )
    h_face_full = 0.5 * (h_cell + h_inf)
    q_species_diff = float(np.sum(J_diff_full * h_face_full))
    q_total = float(q_cond + q_species_diff)

    if not np.isclose(float(np.sum(J_diff_full)), 0.0, rtol=0.0, atol=1.0e-10):
        raise GasFluxError("far-field gas diffusive flux must satisfy sum_i J_i = 0")

    return GasFarfieldFluxPackage(
        face_index=face_index,
        cell_index=cell_index,
        X_face_full=X_face_full,
        Y_face_full=Y_face_full,
        dXdr_full=dXdr_full,
        rho_face=float(rho_face),
        D_face_full=D_face_full,
        Vd0_face_full=Vd0_face_full,
        Vcd_face=Vcd_face,
        J_diff_full=J_diff_full,
        grad_T=float(grad_T),
        k_face=float(k_face),
        h_face_full=h_face_full,
        q_cond=float(q_cond),
        q_species_diff=float(q_species_diff),
        q_total=float(q_total),
    )


__all__ = [
    "GasFaceDiffusionPackage",
    "GasFaceEnergyPackage",
    "GasFarfieldFluxPackage",
    "GasFluxError",
    "build_gas_farfield_boundary_flux_package",
    "build_gas_internal_diffusion_package",
    "build_gas_internal_energy_flux_package",
]
