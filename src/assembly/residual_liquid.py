from __future__ import annotations

"""Liquid bulk residual assembly on the frozen current geometry.

This module assembles liquid bulk residual rows only. It uses current-geometry
backward Euler time terms, continuity-recovered liquid face velocities, and the
single shared InterfaceFacePackage for the last liquid face.

Forbidden here:
- using old-geometry volumes in time terms
- rebuilding old conservative quantities from old primitive variables
- adding a liquid continuity residual row
- updating dot_a or any outer geometry quantity
- recomputing interface properties or interface fluxes
- treating the interface face as an ordinary internal face
- using u_abs instead of u_rel in convective terms
- hard-coding global row offsets instead of layout accessors
"""

from dataclasses import dataclass
from typing import Any

import numpy as np

from core.layout import UnknownLayout
from core.types import Mesh1D, OldStateOnCurrentGeometry, SpeciesMaps, State
from physics.flux_convective import (
    build_liquid_center_boundary_convective_flux,
    build_liquid_internal_convective_flux_scalar,
    build_liquid_internal_convective_flux_vector,
)
from physics.flux_liq import (
    build_liquid_center_boundary_flux,
    build_liquid_internal_diffusion_package,
    build_liquid_internal_energy_flux_package,
)
from physics.interface_face import InterfaceFacePackage
from physics.velocity_recovery import VelocityRecoveryPackage


class LiquidResidualAssemblyError(ValueError):
    """Raised when liquid residual assembly inputs are inconsistent."""


@dataclass(frozen=True)
class LiquidFaceFluxView:
    conv_rhoh: np.ndarray
    conv_rhoY_red: np.ndarray
    q_face: np.ndarray
    J_face_full: np.ndarray
    J_face_red: np.ndarray


@dataclass(frozen=True)
class LiquidConservativeView:
    rhoh_new: np.ndarray
    enthalpy_old_content: np.ndarray
    rhoY_new_red: np.ndarray
    species_mass_old_red: np.ndarray


@dataclass(frozen=True)
class LiquidResidualResult:
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
        raise LiquidResidualAssemblyError(f"{name} must be a 1D float array")
    if not np.all(np.isfinite(arr)):
        raise LiquidResidualAssemblyError(f"{name} must contain only finite values")
    if expected_size is not None and arr.shape[0] != expected_size:
        raise LiquidResidualAssemblyError(f"{name} must have length {expected_size}")
    return arr


def _as_2d_finite_float_array(name: str, value: Any, *, expected_shape: tuple[int, int] | None = None) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim != 2:
        raise LiquidResidualAssemblyError(f"{name} must be a 2D float array")
    if not np.all(np.isfinite(arr)):
        raise LiquidResidualAssemblyError(f"{name} must contain only finite values")
    if expected_shape is not None and arr.shape != expected_shape:
        raise LiquidResidualAssemblyError(f"{name} must have shape {expected_shape}")
    return arr


def _normalize_owned_liq_cells(mesh: Mesh1D, owned_liq_cells: Any | None) -> np.ndarray:
    if owned_liq_cells is None:
        return np.arange(mesh.n_liq, dtype=np.int64)
    owned = np.asarray(owned_liq_cells, dtype=np.int64)
    if owned.ndim != 1:
        raise LiquidResidualAssemblyError("owned_liq_cells must be a 1D integer array")
    if owned.size == 0:
        return np.zeros(0, dtype=np.int64)
    if np.any(owned < 0) or np.any(owned >= mesh.n_liq):
        raise LiquidResidualAssemblyError("owned_liq_cells contains out-of-range liquid cell indices")
    if np.unique(owned).size != owned.size:
        raise LiquidResidualAssemblyError("owned_liq_cells must not contain duplicates")
    return owned


def _validate_inputs(
    *,
    state_trial: State,
    old_state_current_geom: OldStateOnCurrentGeometry,
    props_trial: Any,
    mesh: Mesh1D,
    layout: UnknownLayout,
    species_maps: SpeciesMaps,
    liquid_velocity: VelocityRecoveryPackage,
    iface_face_pkg: InterfaceFacePackage,
    owned_liq_cells: np.ndarray,
) -> None:
    if not old_state_current_geom.mesh.same_geometry(mesh):
        raise LiquidResidualAssemblyError("old_state_current_geom.mesh must match current mesh exactly")
    if mesh.n_liq != state_trial.n_liq_cells:
        raise LiquidResidualAssemblyError("mesh.n_liq must match state_trial.n_liq_cells")
    if mesh.n_liq != layout.n_liq_cells:
        raise LiquidResidualAssemblyError("mesh.n_liq must match layout.n_liq_cells")
    if state_trial.n_liq_species_full != species_maps.n_liq_full:
        raise LiquidResidualAssemblyError("state_trial liquid species dimension must match species_maps.n_liq_full")
    if old_state_current_geom.contents.mass_l.shape[0] != mesh.n_liq:
        raise LiquidResidualAssemblyError("old_state_current_geom.contents.mass_l must match liquid cell count")
    if old_state_current_geom.contents.enthalpy_l.shape[0] != mesh.n_liq:
        raise LiquidResidualAssemblyError("old_state_current_geom.contents.enthalpy_l must match liquid cell count")
    if old_state_current_geom.contents.species_mass_l.shape != (mesh.n_liq, species_maps.n_liq_full):
        raise LiquidResidualAssemblyError(
            "old_state_current_geom.contents.species_mass_l must have shape (n_liq, n_liq_full)"
        )
    Yl_full = _as_2d_finite_float_array(
        "state_trial.Yl_full",
        state_trial.Yl_full,
        expected_shape=(mesh.n_liq, species_maps.n_liq_full),
    )
    if np.any(Yl_full < 0.0):
        raise LiquidResidualAssemblyError("state_trial.Yl_full must be non-negative")
    if liquid_velocity.liquid.u_face_rel.shape[0] != mesh.n_liq + 1:
        raise LiquidResidualAssemblyError("liquid_velocity.liquid.u_face_rel must have length n_liq + 1")
    if liquid_velocity.liquid.u_face_abs.shape[0] != mesh.n_liq + 1:
        raise LiquidResidualAssemblyError("liquid_velocity.liquid.u_face_abs must have length n_liq + 1")
    if liquid_velocity.liquid.vc_face.shape[0] != mesh.n_liq + 1:
        raise LiquidResidualAssemblyError("liquid_velocity.liquid.vc_face must have length n_liq + 1")
    if liquid_velocity.liquid.area_face.shape[0] != mesh.n_liq + 1:
        raise LiquidResidualAssemblyError("liquid_velocity.liquid.area_face must have length n_liq + 1")
    if not np.allclose(
        liquid_velocity.liquid.u_face_rel,
        liquid_velocity.liquid.u_face_abs - liquid_velocity.liquid.vc_face,
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise LiquidResidualAssemblyError("liquid_velocity.liquid.u_face_rel must equal u_face_abs - vc_face")
    if layout.n_liq_red != species_maps.n_liq_red:
        raise LiquidResidualAssemblyError("layout.n_liq_red must match species_maps.n_liq_red")
    if layout.has_liq_species_bulk and layout.n_liq_red <= 0:
        raise LiquidResidualAssemblyError("layout.has_liq_species_bulk implies layout.n_liq_red > 0")
    _as_1d_float_array("props_trial.rho_l", props_trial.rho_l, expected_size=mesh.n_liq)
    _as_1d_float_array("props_trial.h_l", props_trial.h_l, expected_size=mesh.n_liq)
    _as_1d_float_array("old_state_current_geom.contents.enthalpy_l", old_state_current_geom.contents.enthalpy_l, expected_size=mesh.n_liq)
    _as_2d_finite_float_array(
        "old_state_current_geom.contents.species_mass_l",
        old_state_current_geom.contents.species_mass_l,
        expected_shape=(mesh.n_liq, species_maps.n_liq_full),
    )
    iface_Ys_l = _as_1d_float_array("iface_face_pkg.Ys_l_full", iface_face_pkg.Ys_l_full, expected_size=species_maps.n_liq_full)
    if np.any(iface_Ys_l < 0.0):
        raise LiquidResidualAssemblyError("iface_face_pkg.Ys_l_full must be non-negative")
    _as_1d_float_array("iface_face_pkg.J_l_full", iface_face_pkg.J_l_full, expected_size=species_maps.n_liq_full)
    _as_1d_float_array("iface_face_pkg.N_l_full", iface_face_pkg.N_l_full, expected_size=species_maps.n_liq_full)
    if iface_face_pkg.J_l_full.shape[0] != species_maps.n_liq_full:
        raise LiquidResidualAssemblyError("iface_face_pkg.J_l_full must use full liquid species ordering")
    if not np.isfinite(iface_face_pkg.rho_s_l):
        raise LiquidResidualAssemblyError("iface_face_pkg.rho_s_l must be finite")
    if not np.isfinite(iface_face_pkg.h_s_l):
        raise LiquidResidualAssemblyError("iface_face_pkg.h_s_l must be finite")
    if not np.isfinite(iface_face_pkg.q_l_s):
        raise LiquidResidualAssemblyError("iface_face_pkg.q_l_s must be finite")
    if not np.isfinite(iface_face_pkg.E_l_s):
        raise LiquidResidualAssemblyError("iface_face_pkg.E_l_s must be finite")
    if not np.isclose(
        float(mesh.face_areas[mesh.interface_face_index]),
        float(liquid_velocity.liquid.area_face[-1]),
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise LiquidResidualAssemblyError("interface face area must match liquid velocity face area")
    _ = owned_liq_cells


def _validate_internal_face_package_alignment(
    *,
    diff_face_indices: np.ndarray,
    energy_face_indices: np.ndarray,
    conv_h_face_indices: np.ndarray,
    conv_y_face_indices: np.ndarray | None,
) -> None:
    if not np.array_equal(energy_face_indices, diff_face_indices):
        raise LiquidResidualAssemblyError("liquid internal energy package face ordering must match diffusion package")
    if not np.array_equal(conv_h_face_indices, diff_face_indices):
        raise LiquidResidualAssemblyError("liquid internal scalar convective package face ordering must match diffusion package")
    if conv_y_face_indices is not None and not np.array_equal(conv_y_face_indices, diff_face_indices):
        raise LiquidResidualAssemblyError("liquid internal vector convective package face ordering must match diffusion package")


def _build_liquid_conservative_view(
    *,
    state_trial: State,
    old_state_current_geom: OldStateOnCurrentGeometry,
    props_trial: Any,
    species_maps: SpeciesMaps,
) -> LiquidConservativeView:
    rho_l = _as_1d_float_array("props_trial.rho_l", props_trial.rho_l, expected_size=state_trial.n_liq_cells)
    h_l = _as_1d_float_array("props_trial.h_l", props_trial.h_l, expected_size=state_trial.n_liq_cells)
    rhoh_new = rho_l * h_l
    enthalpy_old_content = _as_1d_float_array(
        "old_state_current_geom.contents.enthalpy_l",
        old_state_current_geom.contents.enthalpy_l,
        expected_size=state_trial.n_liq_cells,
    )

    rhoY_new_full = rho_l[:, None] * state_trial.Yl_full
    red_to_full = np.asarray(species_maps.liq_reduced_to_full, dtype=np.int64)
    if red_to_full.size == 0:
        rhoY_new_red = np.zeros((0, state_trial.n_liq_cells), dtype=np.float64)
        species_mass_old_red = np.zeros((0, state_trial.n_liq_cells), dtype=np.float64)
    else:
        rhoY_new_red = rhoY_new_full[:, red_to_full].T
        species_mass_old_red = old_state_current_geom.contents.species_mass_l[:, red_to_full].T

    return LiquidConservativeView(
        rhoh_new=rhoh_new,
        enthalpy_old_content=np.asarray(enthalpy_old_content, dtype=np.float64),
        rhoY_new_red=np.asarray(rhoY_new_red, dtype=np.float64),
        species_mass_old_red=np.asarray(species_mass_old_red, dtype=np.float64),
    )


def _build_liquid_face_flux_view(
    *,
    state_trial: State,
    mesh: Mesh1D,
    species_maps: SpeciesMaps,
    liquid_thermo: Any,
    props_trial: Any,
    liquid_velocity: VelocityRecoveryPackage,
    iface_face_pkg: InterfaceFacePackage,
) -> LiquidFaceFluxView:
    n_liq = mesh.n_liq
    n_full = species_maps.n_liq_full
    n_red = species_maps.n_liq_red
    red_to_full = np.asarray(species_maps.liq_reduced_to_full, dtype=np.int64)

    conv_rhoh = np.zeros(n_liq + 1, dtype=np.float64)
    conv_rhoY_red = np.zeros((n_red, n_liq + 1), dtype=np.float64)
    q_face = np.zeros(n_liq + 1, dtype=np.float64)
    J_face_full = np.zeros((n_full, n_liq + 1), dtype=np.float64)

    rho_l = _as_1d_float_array("props_trial.rho_l", props_trial.rho_l, expected_size=n_liq)
    h_l = _as_1d_float_array("props_trial.h_l", props_trial.h_l, expected_size=n_liq)
    rhoh_cell = rho_l * h_l
    area_if = float(mesh.face_areas[mesh.interface_face_index])
    rhoY_red_cell = np.zeros((n_liq, n_red), dtype=np.float64)
    if n_red > 0:
        rhoY_full = rho_l[:, None] * state_trial.Yl_full
        rhoY_red_cell = rhoY_full[:, red_to_full]

    center_J_full, center_q = build_liquid_center_boundary_flux(n_full)
    J_face_full[:, 0] = center_J_full
    q_face[0] = center_q
    center_conv_h = build_liquid_center_boundary_convective_flux(0.0)
    conv_rhoh[0] = float(center_conv_h.flux[0])
    if n_red > 0:
        center_conv_y = build_liquid_center_boundary_convective_flux(np.zeros(n_red, dtype=np.float64))
        conv_rhoY_red[:, 0] = center_conv_y.flux

    liquid_closure_index = None
    if species_maps.liq_closure_name is not None:
        liquid_closure_index = species_maps.liq_full_names.index(species_maps.liq_closure_name)

    diff_pkg = build_liquid_internal_diffusion_package(
        mesh,
        state_trial,
        liquid_thermo,
        liquid_closure_index=liquid_closure_index,
    )
    energy_pkg = build_liquid_internal_energy_flux_package(mesh, state_trial, liquid_thermo, diff_pkg)
    conv_h_pkg = build_liquid_internal_convective_flux_scalar(
        mesh,
        rhoh_cell,
        liquid_velocity.liquid.u_face_abs,
        liquid_velocity.liquid.vc_face,
    )

    if n_red > 0:
        conv_y_pkg = build_liquid_internal_convective_flux_vector(
            mesh,
            rhoY_red_cell,
            liquid_velocity.liquid.u_face_abs,
            liquid_velocity.liquid.vc_face,
        )
    else:
        conv_y_pkg = None

    _validate_internal_face_package_alignment(
        diff_face_indices=diff_pkg.face_indices,
        energy_face_indices=energy_pkg.face_indices,
        conv_h_face_indices=conv_h_pkg.face_indices,
        conv_y_face_indices=None if conv_y_pkg is None else conv_y_pkg.face_indices,
    )

    for local_idx, face_idx in enumerate(diff_pkg.face_indices.tolist()):
        J_face_full[:, face_idx] = diff_pkg.J_diff_full[local_idx, :]
        q_face[face_idx] = energy_pkg.q_total[local_idx]
        conv_rhoh[face_idx] = conv_h_pkg.flux[local_idx]
        if n_red > 0 and conv_y_pkg is not None:
            conv_rhoY_red[:, face_idx] = conv_y_pkg.flux[local_idx, :]

    # Keep the last liquid face single-sourced from InterfaceFacePackage.
    conv_rhoh[-1] = (float(iface_face_pkg.E_l_s) - float(iface_face_pkg.q_l_s)) * area_if
    if n_red > 0:
        N_l_red_if = _as_1d_float_array("iface_face_pkg.N_l_full", iface_face_pkg.N_l_full, expected_size=n_full)[red_to_full]
        J_l_red_if = _as_1d_float_array("iface_face_pkg.J_l_full", iface_face_pkg.J_l_full, expected_size=n_full)[red_to_full]
        conv_rhoY_red[:, -1] = (N_l_red_if - J_l_red_if) * area_if
    q_face[-1] = float(iface_face_pkg.q_l_s)
    J_face_full[:, -1] = _as_1d_float_array("iface_face_pkg.J_l_full", iface_face_pkg.J_l_full, expected_size=n_full)

    if n_red == 0:
        J_face_red = np.zeros((0, n_liq + 1), dtype=np.float64)
    else:
        J_face_red = J_face_full[red_to_full, :]

    return LiquidFaceFluxView(
        conv_rhoh=conv_rhoh,
        conv_rhoY_red=conv_rhoY_red,
        q_face=q_face,
        J_face_full=J_face_full,
        J_face_red=J_face_red,
    )


def _assemble_liquid_energy_rows(
    *,
    mesh: Mesh1D,
    layout: UnknownLayout,
    dt: float,
    conservative: LiquidConservativeView,
    flux_view: LiquidFaceFluxView,
    owned_liq_cells: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    volumes = _as_1d_float_array("liquid cell volumes", mesh.volumes[mesh.liq_slice], expected_size=mesh.n_liq)
    areas = _as_1d_float_array("liquid face areas", mesh.face_areas[: mesh.n_liq + 1], expected_size=mesh.n_liq + 1)

    time_energy_full = (conservative.rhoh_new * volumes - conservative.enthalpy_old_content) / dt
    conv_energy = np.empty(owned_liq_cells.size, dtype=np.float64)
    diff_energy = np.empty(owned_liq_cells.size, dtype=np.float64)
    time_energy = np.empty(owned_liq_cells.size, dtype=np.float64)
    energy_values = np.empty(owned_liq_cells.size, dtype=np.float64)
    energy_rows = np.empty(owned_liq_cells.size, dtype=np.int64)

    for local_idx, n in enumerate(owned_liq_cells.tolist()):
        f_left = n
        f_right = n + 1
        time_energy[local_idx] = time_energy_full[n]
        conv_energy[local_idx] = flux_view.conv_rhoh[f_right] - flux_view.conv_rhoh[f_left]
        diff_energy[local_idx] = flux_view.q_face[f_right] * areas[f_right] - flux_view.q_face[f_left] * areas[f_left]
        energy_values[local_idx] = time_energy[local_idx] + conv_energy[local_idx] + diff_energy[local_idx]
        energy_rows[local_idx] = layout.liq_temperature_index(n)

    return energy_rows, energy_values, time_energy, conv_energy, diff_energy


def _assemble_liquid_species_rows(
    *,
    mesh: Mesh1D,
    layout: UnknownLayout,
    dt: float,
    conservative: LiquidConservativeView,
    flux_view: LiquidFaceFluxView,
    owned_liq_cells: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_red = layout.n_liq_red
    if n_red == 0:
        empty_i = np.zeros(0, dtype=np.int64)
        empty_f = np.zeros(0, dtype=np.float64)
        empty_terms = np.zeros((0, owned_liq_cells.size), dtype=np.float64)
        return empty_i, empty_f, empty_terms, empty_terms, empty_terms

    volumes = _as_1d_float_array("liquid cell volumes", mesh.volumes[mesh.liq_slice], expected_size=mesh.n_liq)
    areas = _as_1d_float_array("liquid face areas", mesh.face_areas[: mesh.n_liq + 1], expected_size=mesh.n_liq + 1)

    time_species = np.empty((n_red, owned_liq_cells.size), dtype=np.float64)
    conv_species = np.empty((n_red, owned_liq_cells.size), dtype=np.float64)
    diff_species = np.empty((n_red, owned_liq_cells.size), dtype=np.float64)
    rows: list[int] = []
    vals: list[float] = []

    for local_idx, n in enumerate(owned_liq_cells.tolist()):
        f_left = n
        f_right = n + 1
        species_slice = layout.liq_species_slice_for_cell(n)
        if species_slice.stop - species_slice.start != n_red:
            raise LiquidResidualAssemblyError("layout liquid species slice length must match layout.n_liq_red")
        time_species[:, local_idx] = (
            conservative.rhoY_new_red[:, n] * volumes[n]
            - conservative.species_mass_old_red[:, n]
        ) / dt
        conv_species[:, local_idx] = flux_view.conv_rhoY_red[:, f_right] - flux_view.conv_rhoY_red[:, f_left]
        diff_species[:, local_idx] = (
            flux_view.J_face_red[:, f_right] * areas[f_right] - flux_view.J_face_red[:, f_left] * areas[f_left]
        )
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


def assemble_liquid_residual(
    *,
    state_trial: State,
    old_state_current_geom: OldStateOnCurrentGeometry,
    props_trial: Any,
    mesh: Mesh1D,
    layout: UnknownLayout,
    species_maps: SpeciesMaps,
    liquid_thermo: Any,
    liquid_velocity: VelocityRecoveryPackage,
    iface_face_pkg: InterfaceFacePackage,
    owned_liq_cells: Any | None = None,
) -> LiquidResidualResult:
    """Assemble liquid bulk residual rows on the current frozen geometry."""

    owned = _normalize_owned_liq_cells(mesh, owned_liq_cells)
    _validate_inputs(
        state_trial=state_trial,
        old_state_current_geom=old_state_current_geom,
        props_trial=props_trial,
        mesh=mesh,
        layout=layout,
        species_maps=species_maps,
        liquid_velocity=liquid_velocity,
        iface_face_pkg=iface_face_pkg,
        owned_liq_cells=owned,
    )
    dt = float(old_state_current_geom.geometry.dt)
    if not np.isfinite(dt) or dt <= 0.0:
        raise LiquidResidualAssemblyError("old_state_current_geom.geometry.dt must be finite and > 0")

    conservative = _build_liquid_conservative_view(
        state_trial=state_trial,
        old_state_current_geom=old_state_current_geom,
        props_trial=props_trial,
        species_maps=species_maps,
    )
    flux_view = _build_liquid_face_flux_view(
        state_trial=state_trial,
        mesh=mesh,
        species_maps=species_maps,
        liquid_thermo=liquid_thermo,
        props_trial=props_trial,
        liquid_velocity=liquid_velocity,
        iface_face_pkg=iface_face_pkg,
    )

    energy_rows, energy_values, time_energy, conv_energy, diff_energy = _assemble_liquid_energy_rows(
        mesh=mesh,
        layout=layout,
        dt=dt,
        conservative=conservative,
        flux_view=flux_view,
        owned_liq_cells=owned,
    )
    species_rows, species_values, time_species, conv_species, diff_species = _assemble_liquid_species_rows(
        mesh=mesh,
        layout=layout,
        dt=dt,
        conservative=conservative,
        flux_view=flux_view,
        owned_liq_cells=owned,
    )

    rows_global = np.concatenate([energy_rows, species_rows])
    values = np.concatenate([energy_values, species_values])
    return LiquidResidualResult(
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
    "LiquidConservativeView",
    "LiquidFaceFluxView",
    "LiquidResidualAssemblyError",
    "LiquidResidualResult",
    "assemble_liquid_residual",
]
