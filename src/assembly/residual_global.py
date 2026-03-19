from __future__ import annotations

"""Global residual orchestration for the fixed-geometry inner solve.

This module coordinates bulk property aggregation, interface-package
construction, continuity-based velocity recovery, and block residual assembly
for the liquid / interface / gas blocks. It must not recompute a second
interface truth source or introduce any outer-geometry updates.

Forbidden here:
- recomputing an alternative interface state inconsistent with physics packages
- bypassing residual_liquid / residual_interface / residual_gas and hand-writing
  block residuals in this file
- treating a dense global numpy residual vector as the primary truth source
- updating outer geometry or dot_a
- introducing a second gas-side Eq.(2.18) residual row
"""

from dataclasses import dataclass
from typing import Any

import numpy as np

from assembly.residual_gas import GasFarFieldBC, GasResidualResult, assemble_gas_residual
from assembly.residual_interface import InterfaceResidualResult, assemble_interface_residual
from assembly.residual_liquid import LiquidResidualResult, assemble_liquid_residual
from core.layout import UnknownLayout
from core.state_pack import apply_trial_vector_to_state
from core.types import (
    ControlSurfaceMetrics,
    Mesh1D,
    OldStateOnCurrentGeometry,
    SpeciesMaps,
    State,
)
from physics.interface_energy import InterfaceEnergyResidualPackage, build_interface_energy_residual_package
from physics.interface_face import InterfaceFacePackage, build_interface_face_package
from physics.interface_mass import InterfaceMassResidualPackage, build_interface_mass_residual_package
from physics.velocity_recovery import VelocityRecoveryPackage, build_velocity_recovery_package
from properties.aggregator import BulkProps, build_bulk_props, validate_state_grid_compatibility


class GlobalResidualAssemblyError(ValueError):
    """Raised when global residual orchestration inputs are inconsistent."""


@dataclass(frozen=True)
class ResidualOwnership:
    owned_liq_cells: np.ndarray
    owned_gas_cells: np.ndarray
    interface_owner_active: bool


@dataclass(frozen=True)
class GlobalResidualResult:
    rows_global: np.ndarray
    values: np.ndarray

    liquid: LiquidResidualResult
    interface: InterfaceResidualResult
    gas: GasResidualResult

    diagnostics: dict[str, Any]


def _as_1d_int_array(name: str, value: Any) -> np.ndarray:
    arr = np.asarray(value, dtype=np.int64)
    if arr.ndim != 1:
        raise GlobalResidualAssemblyError(f"{name} must be a 1D integer array")
    return arr


def _as_1d_float_array(name: str, value: Any) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim != 1:
        raise GlobalResidualAssemblyError(f"{name} must be a 1D float array")
    if not np.all(np.isfinite(arr)):
        raise GlobalResidualAssemblyError(f"{name} must contain only finite values")
    return arr


def _normalize_ownership(mesh: Mesh1D, ownership: ResidualOwnership) -> ResidualOwnership:
    owned_liq = _as_1d_int_array("ownership.owned_liq_cells", ownership.owned_liq_cells)
    owned_gas = _as_1d_int_array("ownership.owned_gas_cells", ownership.owned_gas_cells)
    if owned_liq.size > 0:
        if np.any(owned_liq < 0) or np.any(owned_liq >= mesh.n_liq):
            raise GlobalResidualAssemblyError("ownership.owned_liq_cells contains out-of-range liquid cell indices")
        if np.unique(owned_liq).size != owned_liq.size:
            raise GlobalResidualAssemblyError("ownership.owned_liq_cells must not contain duplicates")
    if owned_gas.size > 0:
        if np.any(owned_gas < 0) or np.any(owned_gas >= mesh.n_gas):
            raise GlobalResidualAssemblyError("ownership.owned_gas_cells contains out-of-range gas cell indices")
        if np.unique(owned_gas).size != owned_gas.size:
            raise GlobalResidualAssemblyError("ownership.owned_gas_cells must not contain duplicates")
    return ResidualOwnership(
        owned_liq_cells=owned_liq.copy(),
        owned_gas_cells=owned_gas.copy(),
        interface_owner_active=bool(ownership.interface_owner_active),
    )


def _normalize_farfield_bc(farfield_bc: Any, *, n_gas_full: int, default_p_inf: float) -> GasFarFieldBC:
    if isinstance(farfield_bc, GasFarFieldBC):
        bc = farfield_bc
    else:
        if not hasattr(farfield_bc, "T_inf") or not hasattr(farfield_bc, "Yg_inf_full"):
            raise GlobalResidualAssemblyError("farfield_bc must provide T_inf and Yg_inf_full")
        p_inf = getattr(farfield_bc, "p_inf", default_p_inf)
        bc = GasFarFieldBC(T_inf=float(farfield_bc.T_inf), Yg_inf_full=np.asarray(farfield_bc.Yg_inf_full, dtype=np.float64), p_inf=float(p_inf))

    Y_inf = _as_1d_float_array("farfield_bc.Yg_inf_full", bc.Yg_inf_full)
    if Y_inf.shape[0] != n_gas_full:
        raise GlobalResidualAssemblyError("farfield_bc.Yg_inf_full length must match species_maps.n_gas_full")
    if np.any(Y_inf < 0.0):
        raise GlobalResidualAssemblyError("farfield_bc.Yg_inf_full must be non-negative")
    if not np.isclose(float(np.sum(Y_inf)), 1.0, rtol=0.0, atol=1.0e-12):
        raise GlobalResidualAssemblyError("farfield_bc.Yg_inf_full must sum to 1")
    if not np.isfinite(float(bc.T_inf)) or float(bc.T_inf) <= 0.0:
        raise GlobalResidualAssemblyError("farfield_bc.T_inf must be finite and > 0")
    if not np.isfinite(float(bc.p_inf)) or float(bc.p_inf) <= 0.0:
        raise GlobalResidualAssemblyError("farfield_bc.p_inf must be finite and > 0")
    return GasFarFieldBC(T_inf=float(bc.T_inf), Yg_inf_full=Y_inf.copy(), p_inf=float(bc.p_inf))


def _validate_global_inputs(
    *,
    state_trial: State,
    old_state_current_geom: OldStateOnCurrentGeometry,
    mesh: Mesh1D,
    layout: UnknownLayout,
    species_maps: SpeciesMaps,
    ownership: ResidualOwnership,
    run_cfg: Any,
    liquid_thermo: Any,
    gas_thermo: Any,
    control_surface_metrics: ControlSurfaceMetrics,
    farfield_bc: GasFarFieldBC,
) -> ResidualOwnership:
    validate_state_grid_compatibility(state_trial, mesh)
    if not old_state_current_geom.mesh.same_geometry(mesh):
        raise GlobalResidualAssemblyError("old_state_current_geom.mesh must match current mesh exactly")
    if mesh.n_liq != layout.n_liq_cells or mesh.n_gas != layout.n_gas_cells:
        raise GlobalResidualAssemblyError("mesh cell counts must match layout cell counts")
    if state_trial.n_liq_species_full != species_maps.n_liq_full:
        raise GlobalResidualAssemblyError("state_trial liquid species count must match species_maps.n_liq_full")
    if state_trial.n_gas_species_full != species_maps.n_gas_full:
        raise GlobalResidualAssemblyError("state_trial gas species count must match species_maps.n_gas_full")
    if not hasattr(run_cfg, "species_maps"):
        raise GlobalResidualAssemblyError("run_cfg must provide species_maps")
    if run_cfg.species_maps is not species_maps:
        raise GlobalResidualAssemblyError("run_cfg.species_maps must be the same SpeciesMaps object used by assembly")
    if not hasattr(run_cfg, "pressure"):
        raise GlobalResidualAssemblyError("run_cfg must provide normalized pressure")
    if liquid_thermo is None or gas_thermo is None:
        raise GlobalResidualAssemblyError("liquid_thermo and gas_thermo must be provided")
    if control_surface_metrics.v_c_faces.shape[0] != mesh.n_faces:
        raise GlobalResidualAssemblyError("control_surface_metrics.v_c_faces length must match mesh.n_faces")
    if control_surface_metrics.interface_face_index != mesh.interface_face_index:
        raise GlobalResidualAssemblyError("control_surface_metrics.interface_face_index must match mesh.interface_face_index")
    if np.any(~np.isfinite(control_surface_metrics.v_c_faces)):
        raise GlobalResidualAssemblyError("control_surface_metrics.v_c_faces must be finite")
    _ = farfield_bc
    return _normalize_ownership(mesh, ownership)


def _state_with_trial_props(state_trial: State, props_trial: BulkProps) -> State:
    state_with_props = state_trial.copy_shallow()
    state_with_props.rho_l = props_trial.rho_l.copy()
    state_with_props.rho_g = props_trial.rho_g.copy()
    state_with_props.hl = props_trial.h_l.copy()
    state_with_props.hg = props_trial.h_g.copy()
    return state_with_props


def _build_trial_properties(
    *,
    state_trial: State,
    mesh: Mesh1D,
    run_cfg: Any,
    liquid_thermo: Any,
    gas_thermo: Any,
) -> tuple[BulkProps, State]:
    props_trial = build_bulk_props(
        state=state_trial,
        grid=mesh,
        liquid_thermo=liquid_thermo,
        gas_thermo=gas_thermo,
        gas_pressure=float(run_cfg.pressure),
    )
    return props_trial, _state_with_trial_props(state_trial, props_trial)


def _build_interface_packages(
    *,
    run_cfg: Any,
    iface_face_pkg: InterfaceFacePackage,
    u_l_if_abs: float,
    u_g_if_abs: float,
) -> tuple[InterfaceFacePackage, InterfaceMassResidualPackage, InterfaceEnergyResidualPackage]:
    iface_mass_pkg = build_interface_mass_residual_package(
        run_cfg=run_cfg,
        iface_pkg=iface_face_pkg,
        u_l_if_abs=float(u_l_if_abs),
        u_g_if_abs=float(u_g_if_abs),
    )
    iface_energy_pkg = build_interface_energy_residual_package(iface_face_pkg)
    return iface_face_pkg, iface_mass_pkg, iface_energy_pkg


def _build_velocity_package(
    *,
    mesh: Mesh1D,
    state_trial: State,
    old_state_current_geom: OldStateOnCurrentGeometry,
    iface_face_pkg: InterfaceFacePackage,
    control_surface_metrics: ControlSurfaceMetrics,
) -> VelocityRecoveryPackage:
    v_c_faces = _as_1d_float_array("control_surface_metrics.v_c_faces", control_surface_metrics.v_c_faces)
    vc_face_liq = v_c_faces[: mesh.n_liq + 1]
    vc_face_gas = v_c_faces[mesh.interface_face_index :]
    return build_velocity_recovery_package(
        mesh=mesh,
        state=state_trial,
        old_mass_on_current_geometry=old_state_current_geom,
        iface_pkg=iface_face_pkg,
        vc_face_liq=vc_face_liq,
        vc_face_gas=vc_face_gas,
        dt=float(old_state_current_geom.geometry.dt),
    )


def _collect_global_rows_values(
    liquid: LiquidResidualResult,
    interface: InterfaceResidualResult,
    gas: GasResidualResult,
) -> tuple[np.ndarray, np.ndarray]:
    rows_global = np.concatenate([liquid.rows_global, interface.rows_global, gas.rows_global])
    values = np.concatenate([liquid.values, interface.values, gas.values])
    return rows_global, values


def _validate_final_rows(
    *,
    rows_global: np.ndarray,
    values: np.ndarray,
    layout: UnknownLayout,
) -> None:
    rows = _as_1d_int_array("rows_global", rows_global)
    vals = _as_1d_float_array("values", values)
    if rows.shape[0] != vals.shape[0]:
        raise GlobalResidualAssemblyError("rows_global and values must have the same length")
    if np.any(rows < 0) or np.any(rows >= layout.total_size):
        raise GlobalResidualAssemblyError("rows_global contains out-of-range global row indices")
    if np.unique(rows).size != rows.size:
        raise GlobalResidualAssemblyError("rows_global must not contain duplicate global row indices")


def assemble_global_residual(
    *,
    state_trial: State,
    old_state_current_geom: OldStateOnCurrentGeometry,
    mesh: Mesh1D,
    layout: UnknownLayout,
    species_maps: SpeciesMaps,
    ownership: ResidualOwnership,
    run_cfg: Any,
    liquid_thermo: Any,
    gas_thermo: Any,
    equilibrium_model: Any = None,
    control_surface_metrics: ControlSurfaceMetrics,
    farfield_bc: Any,
) -> GlobalResidualResult:
    """Assemble the owned global residual rows from frozen-geometry block assemblers."""

    normalized_bc = _normalize_farfield_bc(farfield_bc, n_gas_full=species_maps.n_gas_full, default_p_inf=float(run_cfg.pressure))
    normalized_ownership = _validate_global_inputs(
        state_trial=state_trial,
        old_state_current_geom=old_state_current_geom,
        mesh=mesh,
        layout=layout,
        species_maps=species_maps,
        ownership=ownership,
        run_cfg=run_cfg,
        liquid_thermo=liquid_thermo,
        gas_thermo=gas_thermo,
        control_surface_metrics=control_surface_metrics,
        farfield_bc=normalized_bc,
    )

    props_trial, state_with_props = _build_trial_properties(
        state_trial=state_trial,
        mesh=mesh,
        run_cfg=run_cfg,
        liquid_thermo=liquid_thermo,
        gas_thermo=gas_thermo,
    )

    iface_face_pkg = build_interface_face_package(
        run_cfg=run_cfg,
        mesh=mesh,
        state=state_with_props,
        gas_props=gas_thermo,
        liquid_props=liquid_thermo,
        equilibrium_model=equilibrium_model,
        dot_a_frozen=float(old_state_current_geom.geometry.dot_a),
    )
    velocity_pkg = _build_velocity_package(
        mesh=mesh,
        state_trial=state_with_props,
        old_state_current_geom=old_state_current_geom,
        iface_face_pkg=iface_face_pkg,
        control_surface_metrics=control_surface_metrics,
    )
    iface_face_pkg, iface_mass_pkg, iface_energy_pkg = _build_interface_packages(
        run_cfg=run_cfg,
        iface_face_pkg=iface_face_pkg,
        u_l_if_abs=float(velocity_pkg.u_l_if_abs),
        u_g_if_abs=float(velocity_pkg.u_g_if_abs),
    )

    liquid = assemble_liquid_residual(
        state_trial=state_with_props,
        old_state_current_geom=old_state_current_geom,
        props_trial=props_trial,
        mesh=mesh,
        layout=layout,
        species_maps=species_maps,
        liquid_thermo=liquid_thermo,
        liquid_velocity=velocity_pkg,
        iface_face_pkg=iface_face_pkg,
        owned_liq_cells=normalized_ownership.owned_liq_cells,
    )
    interface = assemble_interface_residual(
        layout=layout,
        species_maps=species_maps,
        iface_face_pkg=iface_face_pkg,
        iface_mass_pkg=iface_mass_pkg,
        iface_energy_pkg=iface_energy_pkg,
        owner_active=normalized_ownership.interface_owner_active,
    )
    gas = assemble_gas_residual(
        state_trial=state_with_props,
        old_state_current_geom=old_state_current_geom,
        props_trial=props_trial,
        mesh=mesh,
        layout=layout,
        species_maps=species_maps,
        gas_thermo=gas_thermo,
        gas_velocity=velocity_pkg,
        iface_face_pkg=iface_face_pkg,
        farfield_bc=normalized_bc,
        owned_gas_cells=normalized_ownership.owned_gas_cells,
    )

    rows_global, values = _collect_global_rows_values(liquid, interface, gas)
    _validate_final_rows(rows_global=rows_global, values=values, layout=layout)

    diagnostics = {
        "row_count": {
            "liquid": int(liquid.rows_global.size),
            "interface": int(interface.rows_global.size),
            "gas": int(gas.rows_global.size),
            "total": int(rows_global.size),
        },
        "ownership": {
            "n_liq_cells": int(normalized_ownership.owned_liq_cells.size),
            "n_gas_cells": int(normalized_ownership.owned_gas_cells.size),
            "interface_owner_active": bool(normalized_ownership.interface_owner_active),
        },
        "block_norm_inf": {
            "liquid": float(np.max(np.abs(liquid.values))) if liquid.values.size else 0.0,
            "interface": float(np.max(np.abs(interface.values))) if interface.values.size else 0.0,
            "gas": float(np.max(np.abs(gas.values))) if gas.values.size else 0.0,
        },
    }

    return GlobalResidualResult(
        rows_global=rows_global,
        values=values,
        liquid=liquid,
        interface=interface,
        gas=gas,
        diagnostics=diagnostics,
    )


def assemble_global_residual_from_trial_view(
    *,
    vec_trial: np.ndarray,
    base_state: State,
    old_state_current_geom: OldStateOnCurrentGeometry,
    mesh: Mesh1D,
    layout: UnknownLayout,
    species_maps: SpeciesMaps,
    ownership: ResidualOwnership,
    run_cfg: Any,
    liquid_thermo: Any,
    gas_thermo: Any,
    equilibrium_model: Any = None,
    control_surface_metrics: ControlSurfaceMetrics,
    farfield_bc: Any,
) -> GlobalResidualResult:
    state_trial = apply_trial_vector_to_state(
        base_state=base_state,
        vec_trial=vec_trial,
        layout=layout,
        species_maps=species_maps,
    )
    return assemble_global_residual(
        state_trial=state_trial,
        old_state_current_geom=old_state_current_geom,
        mesh=mesh,
        layout=layout,
        species_maps=species_maps,
        ownership=ownership,
        run_cfg=run_cfg,
        liquid_thermo=liquid_thermo,
        gas_thermo=gas_thermo,
        equilibrium_model=equilibrium_model,
        control_surface_metrics=control_surface_metrics,
        farfield_bc=farfield_bc,
    )


__all__ = [
    "GlobalResidualAssemblyError",
    "GlobalResidualResult",
    "ResidualOwnership",
    "assemble_global_residual",
    "assemble_global_residual_from_trial_view",
]
