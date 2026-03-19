from __future__ import annotations

"""Gas bulk residual assembly on the frozen current geometry.

This module assembles gas bulk residual rows only. It uses current-geometry
backward Euler time terms, continuity-recovered gas face velocities, the single
shared InterfaceFacePackage on the left boundary face, and a dedicated
far-field closure on the outer boundary face.

All transport packages consumed here are interpreted as flux densities.
Assembly owns the face-area multiplication for diffusive / thermal terms.

Forbidden here:
- using old-geometry volumes in time terms
- rebuilding old conservative quantities from old primitive variables
- adding any gas-side Eq.(2.18) interface residual row
- recomputing interface properties or interface fluxes
- treating the interface face as an ordinary internal face
- treating the far-field face as an ordinary internal face
- using u_abs instead of u_rel in convective terms
- hard-coding global row offsets instead of layout accessors
- adding chemistry source terms
"""

from dataclasses import dataclass
from typing import Any

import numpy as np

from core.layout import UnknownLayout
from core.types import Mesh1D, OldStateOnCurrentGeometry, SpeciesMaps, State
from physics.flux_convective import (
    build_gas_farfield_boundary_convective_flux_scalar,
    build_gas_farfield_boundary_convective_flux_vector,
    build_gas_internal_convective_flux_scalar,
    build_gas_internal_convective_flux_vector,
)
from physics.flux_gas import (
    build_gas_farfield_boundary_flux_package,
    build_gas_internal_diffusion_package,
    build_gas_internal_energy_flux_package,
)
from physics.interface_face import InterfaceFacePackage
from physics.velocity_recovery import VelocityRecoveryPackage


class GasResidualAssemblyError(ValueError):
    """Raised when gas residual assembly inputs are inconsistent."""


@dataclass(frozen=True)
class GasFarFieldBC:
    T_inf: float
    Yg_inf_full: np.ndarray
    p_inf: float


@dataclass(frozen=True)
class GasFaceFluxView:
    conv_rhoh: np.ndarray
    conv_rhoY_red: np.ndarray
    q_face: np.ndarray
    J_face_full: np.ndarray
    J_face_red: np.ndarray


@dataclass(frozen=True)
class GasConservativeView:
    rhoh_new: np.ndarray
    enthalpy_old_content: np.ndarray
    rhoY_new_red: np.ndarray
    species_mass_old_red: np.ndarray


@dataclass(frozen=True)
class GasResidualResult:
    rows_global: np.ndarray
    values: np.ndarray

    energy_rows_global: np.ndarray
    energy_values: np.ndarray

    species_rows_global: np.ndarray
    species_values: np.ndarray

    time_energy: np.ndarray
    conv_energy: np.ndarray
    diff_energy: np.ndarray

    time_species: np.ndarray
    conv_species: np.ndarray
    diff_species: np.ndarray


def _as_1d_float_array(name: str, value: Any, *, expected_size: int | None = None) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim != 1:
        raise GasResidualAssemblyError(f"{name} must be a 1D float array")
    if not np.all(np.isfinite(arr)):
        raise GasResidualAssemblyError(f"{name} must contain only finite values")
    if expected_size is not None and arr.shape[0] != expected_size:
        raise GasResidualAssemblyError(f"{name} must have length {expected_size}")
    return arr


def _as_2d_finite_float_array(name: str, value: Any, *, expected_shape: tuple[int, int] | None = None) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim != 2:
        raise GasResidualAssemblyError(f"{name} must be a 2D float array")
    if not np.all(np.isfinite(arr)):
        raise GasResidualAssemblyError(f"{name} must contain only finite values")
    if expected_shape is not None and arr.shape != expected_shape:
        raise GasResidualAssemblyError(f"{name} must have shape {expected_shape}")
    return arr


def _normalize_owned_gas_cells(mesh: Mesh1D, owned_gas_cells: Any | None) -> np.ndarray:
    if owned_gas_cells is None:
        return np.arange(mesh.n_gas, dtype=np.int64)
    owned = np.asarray(owned_gas_cells, dtype=np.int64)
    if owned.ndim != 1:
        raise GasResidualAssemblyError("owned_gas_cells must be a 1D integer array")
    if owned.size == 0:
        return np.zeros(0, dtype=np.int64)
    if np.any(owned < 0) or np.any(owned >= mesh.n_gas):
        raise GasResidualAssemblyError("owned_gas_cells contains out-of-range gas cell indices")
    if np.unique(owned).size != owned.size:
        raise GasResidualAssemblyError("owned_gas_cells must not contain duplicates")
    return owned


def _validate_internal_face_package_alignment(
    *,
    mesh: Mesh1D,
    diff_face_indices: np.ndarray,
    energy_face_indices: np.ndarray,
    conv_h_face_indices: np.ndarray,
    conv_y_face_indices: np.ndarray | None,
) -> None:
    if not np.array_equal(energy_face_indices, diff_face_indices):
        raise GasResidualAssemblyError("gas internal energy package face ordering must match diffusion package")
    if not np.array_equal(conv_h_face_indices, diff_face_indices):
        raise GasResidualAssemblyError("gas internal scalar convective package face ordering must match diffusion package")
    if conv_y_face_indices is not None and not np.array_equal(conv_y_face_indices, diff_face_indices):
        raise GasResidualAssemblyError("gas internal vector convective package face ordering must match diffusion package")
    if np.any(diff_face_indices <= mesh.interface_face_index):
        raise GasResidualAssemblyError("gas internal face packages must not include the interface face")
    if np.any(diff_face_indices >= mesh.interface_face_index + mesh.n_gas):
        raise GasResidualAssemblyError("gas internal face packages must not include the far-field boundary face")


def _validate_gas_species_mapping(species_maps: SpeciesMaps, n_gas_red: int) -> np.ndarray:
    red_to_full = np.asarray(species_maps.gas_reduced_to_full, dtype=np.int64)
    if red_to_full.ndim != 1:
        raise GasResidualAssemblyError("species_maps.gas_reduced_to_full must be a 1D integer array")
    if red_to_full.shape[0] != n_gas_red:
        raise GasResidualAssemblyError("species_maps.gas_reduced_to_full length must match n_gas_red")
    if np.any(red_to_full < 0) or np.any(red_to_full >= species_maps.n_gas_full):
        raise GasResidualAssemblyError("species_maps.gas_reduced_to_full contains out-of-range full-species indices")
    if np.unique(red_to_full).size != red_to_full.size:
        raise GasResidualAssemblyError("species_maps.gas_reduced_to_full must not contain duplicate full-species indices")
    return red_to_full


def _validate_farfield_bc(farfield_bc: GasFarFieldBC, n_gas_full: int) -> GasFarFieldBC:
    T_inf = float(farfield_bc.T_inf)
    p_inf = float(farfield_bc.p_inf)
    if not np.isfinite(T_inf) or T_inf <= 0.0:
        raise GasResidualAssemblyError("farfield_bc.T_inf must be finite and > 0")
    if not np.isfinite(p_inf) or p_inf <= 0.0:
        raise GasResidualAssemblyError("farfield_bc.p_inf must be finite and > 0")
    Y_inf = _as_1d_float_array("farfield_bc.Yg_inf_full", farfield_bc.Yg_inf_full, expected_size=n_gas_full)
    if np.any(Y_inf < 0.0):
        raise GasResidualAssemblyError("farfield_bc.Yg_inf_full must be non-negative")
    if not np.isclose(float(np.sum(Y_inf)), 1.0, rtol=0.0, atol=1.0e-12):
        raise GasResidualAssemblyError("farfield_bc.Yg_inf_full must sum to 1")
    return GasFarFieldBC(T_inf=T_inf, Yg_inf_full=Y_inf.copy(), p_inf=p_inf)


def _validate_inputs(
    *,
    state_trial: State,
    old_state_current_geom: OldStateOnCurrentGeometry,
    props_trial: Any,
    mesh: Mesh1D,
    layout: UnknownLayout,
    species_maps: SpeciesMaps,
    gas_velocity: VelocityRecoveryPackage,
    iface_face_pkg: InterfaceFacePackage,
    farfield_bc: GasFarFieldBC,
    owned_gas_cells: np.ndarray,
) -> None:
    if not old_state_current_geom.mesh.same_geometry(mesh):
        raise GasResidualAssemblyError("old_state_current_geom.mesh must match current mesh exactly")
    if mesh.n_gas != state_trial.n_gas_cells:
        raise GasResidualAssemblyError("mesh.n_gas must match state_trial.n_gas_cells")
    if mesh.n_gas != layout.n_gas_cells:
        raise GasResidualAssemblyError("mesh.n_gas must match layout.n_gas_cells")
    if state_trial.n_gas_species_full != species_maps.n_gas_full:
        raise GasResidualAssemblyError("state_trial gas species dimension must match species_maps.n_gas_full")
    _as_2d_finite_float_array(
        "state_trial.Yg_full",
        state_trial.Yg_full,
        expected_shape=(mesh.n_gas, species_maps.n_gas_full),
    )
    _as_1d_float_array("props_trial.rho_g", props_trial.rho_g, expected_size=mesh.n_gas)
    _as_1d_float_array("props_trial.h_g", props_trial.h_g, expected_size=mesh.n_gas)
    _as_1d_float_array("old_state_current_geom.contents.enthalpy_g", old_state_current_geom.contents.enthalpy_g, expected_size=mesh.n_gas)
    _as_2d_finite_float_array(
        "old_state_current_geom.contents.species_mass_g",
        old_state_current_geom.contents.species_mass_g,
        expected_shape=(mesh.n_gas, species_maps.n_gas_full),
    )
    if gas_velocity.gas.u_face_abs.shape[0] != mesh.n_gas + 1:
        raise GasResidualAssemblyError("gas_velocity.gas.u_face_abs must have length n_gas + 1")
    if gas_velocity.gas.u_face_rel.shape[0] != mesh.n_gas + 1:
        raise GasResidualAssemblyError("gas_velocity.gas.u_face_rel must have length n_gas + 1")
    if gas_velocity.gas.vc_face.shape[0] != mesh.n_gas + 1:
        raise GasResidualAssemblyError("gas_velocity.gas.vc_face must have length n_gas + 1")
    if gas_velocity.gas.area_face.shape[0] != mesh.n_gas + 1:
        raise GasResidualAssemblyError("gas_velocity.gas.area_face must have length n_gas + 1")
    if gas_velocity.gas.G_face_abs.shape[0] != mesh.n_gas + 1:
        raise GasResidualAssemblyError("gas_velocity.gas.G_face_abs must have length n_gas + 1")
    if not np.allclose(
        gas_velocity.gas.u_face_rel,
        gas_velocity.gas.u_face_abs - gas_velocity.gas.vc_face,
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise GasResidualAssemblyError("gas_velocity.gas.u_face_rel must equal u_face_abs - vc_face")
    if not np.isclose(float(gas_velocity.gas.G_face_abs[0]), float(iface_face_pkg.G_g_if_abs), rtol=0.0, atol=1.0e-12):
        raise GasResidualAssemblyError("gas interface mass flux must match iface_face_pkg.G_g_if_abs")
    if not np.isclose(float(mesh.face_areas[mesh.interface_face_index]), float(gas_velocity.gas.area_face[0]), rtol=0.0, atol=1.0e-12):
        raise GasResidualAssemblyError("interface face area must match gas velocity face area")
    if not np.isclose(float(mesh.face_areas[-1]), float(gas_velocity.gas.area_face[-1]), rtol=0.0, atol=1.0e-12):
        raise GasResidualAssemblyError("far-field face area must match gas velocity face area")
    _as_1d_float_array("iface_face_pkg.J_g_full", iface_face_pkg.J_g_full, expected_size=species_maps.n_gas_full)
    _as_1d_float_array("iface_face_pkg.N_g_full", iface_face_pkg.N_g_full, expected_size=species_maps.n_gas_full)
    if not np.isfinite(iface_face_pkg.q_g_s) or not np.isfinite(iface_face_pkg.E_g_s):
        raise GasResidualAssemblyError("iface_face_pkg.q_g_s and iface_face_pkg.E_g_s must be finite")
    if layout.n_gas_red != species_maps.n_gas_red:
        raise GasResidualAssemblyError("layout.n_gas_red must match species_maps.n_gas_red")
    _validate_gas_species_mapping(species_maps, layout.n_gas_red)
    _ = farfield_bc
    _ = owned_gas_cells


def _build_gas_conservative_view(
    *,
    state_trial: State,
    old_state_current_geom: OldStateOnCurrentGeometry,
    props_trial: Any,
    species_maps: SpeciesMaps,
) -> GasConservativeView:
    rho_g = _as_1d_float_array("props_trial.rho_g", props_trial.rho_g, expected_size=state_trial.n_gas_cells)
    h_g = _as_1d_float_array("props_trial.h_g", props_trial.h_g, expected_size=state_trial.n_gas_cells)
    rhoh_new = rho_g * h_g
    enthalpy_old_content = _as_1d_float_array(
        "old_state_current_geom.contents.enthalpy_g",
        old_state_current_geom.contents.enthalpy_g,
        expected_size=state_trial.n_gas_cells,
    )

    rhoY_new_full = rho_g[:, None] * state_trial.Yg_full
    red_to_full = _validate_gas_species_mapping(species_maps, species_maps.n_gas_red)
    rhoY_new_red = rhoY_new_full[:, red_to_full].T
    species_mass_old_red = old_state_current_geom.contents.species_mass_g[:, red_to_full].T

    return GasConservativeView(
        rhoh_new=rhoh_new,
        enthalpy_old_content=enthalpy_old_content,
        rhoY_new_red=np.asarray(rhoY_new_red, dtype=np.float64),
        species_mass_old_red=np.asarray(species_mass_old_red, dtype=np.float64),
    )


def _build_farfield_conservative_boundary_values(
    *,
    gas_thermo: Any,
    farfield_bc: GasFarFieldBC,
    n_gas_full: int,
) -> tuple[float, np.ndarray]:
    Y_inf = _as_1d_float_array("farfield_bc.Yg_inf_full", farfield_bc.Yg_inf_full, expected_size=n_gas_full)
    rho_inf = float(gas_thermo.density_mass(float(farfield_bc.T_inf), Y_inf, float(farfield_bc.p_inf)))
    h_inf = float(gas_thermo.enthalpy_mass(float(farfield_bc.T_inf), Y_inf, float(farfield_bc.p_inf)))
    if not np.isfinite(rho_inf) or rho_inf <= 0.0:
        raise GasResidualAssemblyError("far-field gas density must be finite and strictly positive")
    if not np.isfinite(h_inf):
        raise GasResidualAssemblyError("far-field gas enthalpy must be finite")
    return float(rho_inf * h_inf), rho_inf * Y_inf


def _build_gas_face_flux_view(
    *,
    state_trial: State,
    mesh: Mesh1D,
    species_maps: SpeciesMaps,
    gas_thermo: Any,
    props_trial: Any,
    gas_velocity: VelocityRecoveryPackage,
    iface_face_pkg: InterfaceFacePackage,
    farfield_bc: GasFarFieldBC,
) -> GasFaceFluxView:
    n_gas = mesh.n_gas
    n_full = species_maps.n_gas_full
    n_red = species_maps.n_gas_red
    red_to_full = _validate_gas_species_mapping(species_maps, n_red)

    conv_rhoh = np.zeros(n_gas + 1, dtype=np.float64)
    conv_rhoY_red = np.zeros((n_red, n_gas + 1), dtype=np.float64)
    q_face = np.zeros(n_gas + 1, dtype=np.float64)
    J_face_full = np.zeros((n_full, n_gas + 1), dtype=np.float64)

    rho_g = _as_1d_float_array("props_trial.rho_g", props_trial.rho_g, expected_size=n_gas)
    h_g = _as_1d_float_array("props_trial.h_g", props_trial.h_g, expected_size=n_gas)
    rhoh_cell = rho_g * h_g
    rhoY_full = rho_g[:, None] * state_trial.Yg_full
    rhoY_red_cell = rhoY_full[:, red_to_full]

    area_if = float(mesh.face_areas[mesh.interface_face_index])
    conv_rhoh[0] = (float(iface_face_pkg.E_g_s) - float(iface_face_pkg.q_g_s)) * area_if
    N_g_red_if = _as_1d_float_array("iface_face_pkg.N_g_full", iface_face_pkg.N_g_full, expected_size=n_full)[red_to_full]
    J_g_red_if = _as_1d_float_array("iface_face_pkg.J_g_full", iface_face_pkg.J_g_full, expected_size=n_full)[red_to_full]
    conv_rhoY_red[:, 0] = (N_g_red_if - J_g_red_if) * area_if
    q_face[0] = float(iface_face_pkg.q_g_s)
    J_face_full[:, 0] = _as_1d_float_array("iface_face_pkg.J_g_full", iface_face_pkg.J_g_full, expected_size=n_full)

    diff_pkg = build_gas_internal_diffusion_package(mesh, state_trial, gas_thermo, float(farfield_bc.p_inf))
    energy_pkg = build_gas_internal_energy_flux_package(mesh, state_trial, gas_thermo, float(farfield_bc.p_inf), diff_pkg)
    conv_h_pkg = build_gas_internal_convective_flux_scalar(
        mesh,
        rhoh_cell,
        gas_velocity.gas.u_face_abs,
        gas_velocity.gas.vc_face,
    )
    conv_y_pkg = build_gas_internal_convective_flux_vector(
        mesh,
        rhoY_red_cell,
        gas_velocity.gas.u_face_abs,
        gas_velocity.gas.vc_face,
    )
    _validate_internal_face_package_alignment(
        mesh=mesh,
        diff_face_indices=diff_pkg.face_indices,
        energy_face_indices=energy_pkg.face_indices,
        conv_h_face_indices=conv_h_pkg.face_indices,
        conv_y_face_indices=conv_y_pkg.face_indices,
    )
    for local_idx, global_face_idx in enumerate(diff_pkg.face_indices.tolist()):
        local_face_idx = int(global_face_idx - mesh.interface_face_index)
        J_face_full[:, local_face_idx] = diff_pkg.J_diff_full[local_idx, :]
        q_face[local_face_idx] = energy_pkg.q_total[local_idx]
        conv_rhoh[local_face_idx] = conv_h_pkg.flux[local_idx]
        conv_rhoY_red[:, local_face_idx] = conv_y_pkg.flux[local_idx, :]

    far_pkg = build_gas_farfield_boundary_flux_package(
        mesh,
        state_trial,
        gas_thermo,
        float(farfield_bc.p_inf),
        float(farfield_bc.T_inf),
        farfield_bc.Yg_inf_full,
    )
    rhoh_inf, rhoY_inf_full = _build_farfield_conservative_boundary_values(
        gas_thermo=gas_thermo,
        farfield_bc=farfield_bc,
        n_gas_full=n_full,
    )
    conv_far_h = build_gas_farfield_boundary_convective_flux_scalar(
        mesh,
        float(rhoh_cell[-1]),
        float(rhoh_inf),
        float(gas_velocity.gas.u_face_abs[-1]),
        float(gas_velocity.gas.vc_face[-1]),
    )
    conv_far_y = build_gas_farfield_boundary_convective_flux_vector(
        mesh,
        rhoY_red_cell[-1, :],
        rhoY_inf_full[red_to_full],
        float(gas_velocity.gas.u_face_abs[-1]),
        float(gas_velocity.gas.vc_face[-1]),
    )
    conv_rhoh[-1] = float(conv_far_h.flux[0])
    conv_rhoY_red[:, -1] = conv_far_y.flux
    q_face[-1] = float(far_pkg.q_total)
    J_face_full[:, -1] = _as_1d_float_array("farfield J_diff_full", far_pkg.J_diff_full, expected_size=n_full)
    J_face_red = J_face_full[red_to_full, :]

    return GasFaceFluxView(
        conv_rhoh=conv_rhoh,
        conv_rhoY_red=conv_rhoY_red,
        q_face=q_face,
        J_face_full=J_face_full,
        J_face_red=J_face_red,
    )


def _assemble_gas_energy_rows(
    *,
    mesh: Mesh1D,
    layout: UnknownLayout,
    dt: float,
    conservative: GasConservativeView,
    flux_view: GasFaceFluxView,
    owned_gas_cells: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    volumes = _as_1d_float_array("gas cell volumes", mesh.volumes[mesh.gas_slice], expected_size=mesh.n_gas)
    areas = _as_1d_float_array("gas face areas", mesh.face_areas[mesh.interface_face_index :], expected_size=mesh.n_gas + 1)
    time_energy_full = (conservative.rhoh_new * volumes - conservative.enthalpy_old_content) / dt

    time_energy = np.empty(owned_gas_cells.size, dtype=np.float64)
    conv_energy = np.empty(owned_gas_cells.size, dtype=np.float64)
    diff_energy = np.empty(owned_gas_cells.size, dtype=np.float64)
    values = np.empty(owned_gas_cells.size, dtype=np.float64)
    rows = np.empty(owned_gas_cells.size, dtype=np.int64)

    for local_idx, n in enumerate(owned_gas_cells.tolist()):
        f_left = n
        f_right = n + 1
        time_energy[local_idx] = time_energy_full[n]
        conv_energy[local_idx] = flux_view.conv_rhoh[f_right] - flux_view.conv_rhoh[f_left]
        diff_energy[local_idx] = flux_view.q_face[f_right] * areas[f_right] - flux_view.q_face[f_left] * areas[f_left]
        values[local_idx] = time_energy[local_idx] + conv_energy[local_idx] + diff_energy[local_idx]
        rows[local_idx] = layout.gas_temperature_index(n)

    return rows, values, time_energy, conv_energy, diff_energy


def _assemble_gas_species_rows(
    *,
    mesh: Mesh1D,
    layout: UnknownLayout,
    dt: float,
    conservative: GasConservativeView,
    flux_view: GasFaceFluxView,
    owned_gas_cells: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_red = layout.n_gas_red
    if n_red == 0:
        empty_i = np.zeros(0, dtype=np.int64)
        empty_f = np.zeros(0, dtype=np.float64)
        empty_terms = np.zeros((0, owned_gas_cells.size), dtype=np.float64)
        return empty_i, empty_f, empty_terms, empty_terms, empty_terms

    volumes = _as_1d_float_array("gas cell volumes", mesh.volumes[mesh.gas_slice], expected_size=mesh.n_gas)
    areas = _as_1d_float_array("gas face areas", mesh.face_areas[mesh.interface_face_index :], expected_size=mesh.n_gas + 1)

    time_species = np.empty((n_red, owned_gas_cells.size), dtype=np.float64)
    conv_species = np.empty((n_red, owned_gas_cells.size), dtype=np.float64)
    diff_species = np.empty((n_red, owned_gas_cells.size), dtype=np.float64)
    rows: list[int] = []
    vals: list[float] = []

    for local_idx, n in enumerate(owned_gas_cells.tolist()):
        f_left = n
        f_right = n + 1
        species_slice = layout.gas_species_slice_for_cell(n)
        if species_slice.stop - species_slice.start != n_red:
            raise GasResidualAssemblyError("layout gas species slice length must match layout.n_gas_red")
        time_species[:, local_idx] = (
            conservative.rhoY_new_red[:, n] * volumes[n] - conservative.species_mass_old_red[:, n]
        ) / dt
        conv_species[:, local_idx] = flux_view.conv_rhoY_red[:, f_right] - flux_view.conv_rhoY_red[:, f_left]
        diff_species[:, local_idx] = flux_view.J_face_red[:, f_right] * areas[f_right] - flux_view.J_face_red[:, f_left] * areas[f_left]
        species_vals_cell = time_species[:, local_idx] + conv_species[:, local_idx] + diff_species[:, local_idx]
        rows.extend(range(species_slice.start, species_slice.stop))
        vals.extend(species_vals_cell.tolist())

    return (
        np.asarray(rows, dtype=np.int64),
        np.asarray(vals, dtype=np.float64),
        time_species,
        conv_species,
        diff_species,
    )


def assemble_gas_residual(
    *,
    state_trial: State,
    old_state_current_geom: OldStateOnCurrentGeometry,
    props_trial: Any,
    mesh: Mesh1D,
    layout: UnknownLayout,
    species_maps: SpeciesMaps,
    gas_thermo: Any,
    gas_velocity: VelocityRecoveryPackage,
    iface_face_pkg: InterfaceFacePackage,
    farfield_bc: GasFarFieldBC,
    owned_gas_cells: Any | None = None,
) -> GasResidualResult:
    """Assemble gas bulk residual rows on the current frozen geometry."""

    owned = _normalize_owned_gas_cells(mesh, owned_gas_cells)
    bc = _validate_farfield_bc(farfield_bc, species_maps.n_gas_full)
    _validate_inputs(
        state_trial=state_trial,
        old_state_current_geom=old_state_current_geom,
        props_trial=props_trial,
        mesh=mesh,
        layout=layout,
        species_maps=species_maps,
        gas_velocity=gas_velocity,
        iface_face_pkg=iface_face_pkg,
        farfield_bc=bc,
        owned_gas_cells=owned,
    )
    dt = float(old_state_current_geom.geometry.dt)
    if not np.isfinite(dt) or dt <= 0.0:
        raise GasResidualAssemblyError("old_state_current_geom.geometry.dt must be finite and > 0")

    conservative = _build_gas_conservative_view(
        state_trial=state_trial,
        old_state_current_geom=old_state_current_geom,
        props_trial=props_trial,
        species_maps=species_maps,
    )
    flux_view = _build_gas_face_flux_view(
        state_trial=state_trial,
        mesh=mesh,
        species_maps=species_maps,
        gas_thermo=gas_thermo,
        props_trial=props_trial,
        gas_velocity=gas_velocity,
        iface_face_pkg=iface_face_pkg,
        farfield_bc=bc,
    )

    energy_rows, energy_values, time_energy, conv_energy, diff_energy = _assemble_gas_energy_rows(
        mesh=mesh,
        layout=layout,
        dt=dt,
        conservative=conservative,
        flux_view=flux_view,
        owned_gas_cells=owned,
    )
    species_rows, species_values, time_species, conv_species, diff_species = _assemble_gas_species_rows(
        mesh=mesh,
        layout=layout,
        dt=dt,
        conservative=conservative,
        flux_view=flux_view,
        owned_gas_cells=owned,
    )

    rows_global = np.concatenate([energy_rows, species_rows])
    values = np.concatenate([energy_values, species_values])
    return GasResidualResult(
        rows_global=rows_global,
        values=values,
        energy_rows_global=energy_rows,
        energy_values=energy_values,
        species_rows_global=species_rows,
        species_values=species_values,
        time_energy=time_energy,
        conv_energy=conv_energy,
        diff_energy=diff_energy,
        time_species=time_species,
        conv_species=conv_species,
        diff_species=diff_species,
    )


__all__ = [
    "GasConservativeView",
    "GasFaceFluxView",
    "GasFarFieldBC",
    "GasResidualAssemblyError",
    "GasResidualResult",
    "assemble_gas_residual",
]
