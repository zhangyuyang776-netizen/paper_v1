from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest

import assembly.residual_liquid as residual_liquid_module
from assembly.residual_liquid import assemble_liquid_residual
from core.layout import UnknownLayout
from core.types import (
    ConservativeContents,
    GeometryState,
    InterfaceState,
    Mesh1D,
    OldStateOnCurrentGeometry,
    RegionSlices,
    SpeciesMaps,
    State,
)
from physics.interface_face import InterfaceFacePackage
from physics.velocity_recovery import PhaseVelocityRecovery, VelocityRecoveryPackage
from physics.flux_convective import ConvectiveInternalFluxPackage


@dataclass
class DummyLiquidThermo:
    n_species: int

    def conductivity(self, T: float, Y_full: np.ndarray) -> float:
        return 1.0

    def pure_enthalpy_vector(self, T: float) -> np.ndarray:
        return np.linspace(10.0, 10.0 + self.n_species - 1, self.n_species, dtype=np.float64)

    def diffusivity(self, T: float, Y_full: np.ndarray) -> np.ndarray | None:
        if self.n_species == 1:
            return None
        return np.linspace(1.0e-9, 2.0e-9, self.n_species, dtype=np.float64)


def _make_mesh() -> Mesh1D:
    return Mesh1D(
        r_faces=np.asarray([0.0, 1.0, 2.0, 3.0], dtype=np.float64),
        r_centers=np.asarray([0.5, 1.5, 2.5], dtype=np.float64),
        volumes=np.asarray([1.0, 1.0, 1.0], dtype=np.float64),
        face_areas=np.asarray([0.0, 2.0, 4.0, 6.0], dtype=np.float64),
        dr=np.asarray([1.0, 1.0, 1.0], dtype=np.float64),
        region_slices=RegionSlices(
            liq=slice(0, 2),
            gas_near=slice(2, 3),
            gas_far=slice(3, 3),
            gas_all=slice(2, 3),
        ),
        interface_face_index=2,
        interface_cell_liq=1,
        interface_cell_gas=2,
    )


def _make_species_maps_single() -> SpeciesMaps:
    return SpeciesMaps(
        liq_full_names=("ethanol",),
        liq_active_names=(),
        liq_closure_name="ethanol",
        gas_full_names=("C2H5OH", "N2"),
        gas_active_names=("C2H5OH",),
        gas_closure_name="N2",
        liq_full_to_reduced=np.asarray([-1], dtype=np.int64),
        liq_reduced_to_full=np.asarray([], dtype=np.int64),
        gas_full_to_reduced=np.asarray([0, -1], dtype=np.int64),
        gas_reduced_to_full=np.asarray([0], dtype=np.int64),
        liq_full_to_gas_full=np.asarray([0], dtype=np.int64),
    )


def _make_species_maps_multi() -> SpeciesMaps:
    return SpeciesMaps(
        liq_full_names=("ethanol", "water"),
        liq_active_names=("ethanol",),
        liq_closure_name="water",
        gas_full_names=("C2H5OH", "H2O", "N2"),
        gas_active_names=("C2H5OH", "H2O"),
        gas_closure_name="N2",
        liq_full_to_reduced=np.asarray([0, -1], dtype=np.int64),
        liq_reduced_to_full=np.asarray([0], dtype=np.int64),
        gas_full_to_reduced=np.asarray([0, 1, -1], dtype=np.int64),
        gas_reduced_to_full=np.asarray([0, 1], dtype=np.int64),
        liq_full_to_gas_full=np.asarray([0, 1], dtype=np.int64),
    )


def _make_species_maps_three_full() -> SpeciesMaps:
    return SpeciesMaps(
        liq_full_names=("A", "B", "C"),
        liq_active_names=("A", "C"),
        liq_closure_name="B",
        gas_full_names=("GA", "GB", "GC", "N2"),
        gas_active_names=("GA", "GB", "GC"),
        gas_closure_name="N2",
        liq_full_to_reduced=np.asarray([0, -1, 1], dtype=np.int64),
        liq_reduced_to_full=np.asarray([0, 2], dtype=np.int64),
        gas_full_to_reduced=np.asarray([0, 1, 2, -1], dtype=np.int64),
        gas_reduced_to_full=np.asarray([0, 1, 2], dtype=np.int64),
        liq_full_to_gas_full=np.asarray([0, 1, 2], dtype=np.int64),
    )


def _make_layout_single(mesh: Mesh1D) -> UnknownLayout:
    return UnknownLayout(
        unknowns_profile="U_A",
        n_liq_cells=mesh.n_liq,
        n_gas_cells=mesh.n_gas,
        n_liq_red=0,
        n_gas_red=1,
        liq_cell_width=1,
        gas_cell_width=2,
        n_if_unknowns=3,
        liq_block=slice(0, 2),
        if_block=slice(2, 5),
        gas_block=slice(5, 7),
        total_size=7,
        liq_temperature_slice=slice(0, 2, 1),
        liq_species_local_slice=slice(1, 1),
        if_temperature_slice=slice(2, 3),
        if_gas_species_slice=slice(3, 4),
        if_mpp_slice=slice(4, 5),
        if_liq_species_slice=slice(5, 5),
        gas_temperature_slice=slice(5, 7, 2),
        gas_species_local_slice=slice(1, 2),
    )


def _make_layout_multi(mesh: Mesh1D) -> UnknownLayout:
    return UnknownLayout(
        unknowns_profile="U_B",
        n_liq_cells=mesh.n_liq,
        n_gas_cells=mesh.n_gas,
        n_liq_red=1,
        n_gas_red=2,
        liq_cell_width=2,
        gas_cell_width=3,
        n_if_unknowns=5,
        liq_block=slice(0, 4),
        if_block=slice(4, 9),
        gas_block=slice(9, 12),
        total_size=12,
        liq_temperature_slice=slice(0, 4, 2),
        liq_species_local_slice=slice(1, 2),
        if_temperature_slice=slice(4, 5),
        if_gas_species_slice=slice(5, 7),
        if_mpp_slice=slice(7, 8),
        if_liq_species_slice=slice(8, 9),
        gas_temperature_slice=slice(9, 12, 3),
        gas_species_local_slice=slice(1, 3),
    )


def _make_layout_three_full(mesh: Mesh1D) -> UnknownLayout:
    return UnknownLayout(
        unknowns_profile="U_B",
        n_liq_cells=mesh.n_liq,
        n_gas_cells=mesh.n_gas,
        n_liq_red=2,
        n_gas_red=3,
        liq_cell_width=3,
        gas_cell_width=4,
        n_if_unknowns=7,
        liq_block=slice(0, 6),
        if_block=slice(6, 13),
        gas_block=slice(13, 17),
        total_size=17,
        liq_temperature_slice=slice(0, 6, 3),
        liq_species_local_slice=slice(1, 3),
        if_temperature_slice=slice(6, 7),
        if_gas_species_slice=slice(7, 10),
        if_mpp_slice=slice(10, 11),
        if_liq_species_slice=slice(11, 13),
        gas_temperature_slice=slice(13, 17, 4),
        gas_species_local_slice=slice(1, 4),
    )


def _make_state(mesh: Mesh1D, *, Yl_full: np.ndarray, Yg_full: np.ndarray, Tl: np.ndarray | None = None) -> State:
    return State(
        Tl=np.asarray([300.0, 300.0] if Tl is None else Tl, dtype=np.float64),
        Yl_full=np.asarray(Yl_full, dtype=np.float64),
        Tg=np.asarray([320.0], dtype=np.float64),
        Yg_full=np.asarray(Yg_full, dtype=np.float64),
        interface=InterfaceState(
            Ts=300.0,
            mpp=0.0,
            Ys_g_full=np.asarray(Yg_full[0], dtype=np.float64),
            Ys_l_full=np.asarray(Yl_full[-1], dtype=np.float64),
        ),
        rho_l=np.asarray([1.0, 1.0], dtype=np.float64),
        rho_g=np.asarray([1.0], dtype=np.float64),
        hl=np.asarray([10.0, 10.0], dtype=np.float64),
        hg=np.asarray([20.0], dtype=np.float64),
    )


def _make_old_state_current_geom(
    mesh: Mesh1D,
    state: State,
    *,
    enthalpy_l: np.ndarray,
    species_mass_l: np.ndarray,
) -> OldStateOnCurrentGeometry:
    contents = ConservativeContents(
        mass_l=np.asarray([1.0, 1.0], dtype=np.float64),
        species_mass_l=np.asarray(species_mass_l, dtype=np.float64),
        enthalpy_l=np.asarray(enthalpy_l, dtype=np.float64),
        mass_g=np.asarray([1.0], dtype=np.float64),
        species_mass_g=np.asarray(state.Yg_full, dtype=np.float64),
        enthalpy_g=np.asarray([1.0], dtype=np.float64),
    )
    geometry = GeometryState(t=0.0, dt=1.0, a=2.0, dot_a=0.0, r_end=3.0, step_index=0, outer_iter_index=0)
    return OldStateOnCurrentGeometry(contents=contents, state=state, geometry=geometry, mesh=mesh)


def _make_velocity_pkg(mesh: Mesh1D, *, u_abs_liq: np.ndarray, vc_liq: np.ndarray) -> VelocityRecoveryPackage:
    phase_liq = PhaseVelocityRecovery(
        G_face_abs=np.asarray(u_abs_liq, dtype=np.float64),
        rho_face=np.asarray([1.0, 1.0, 1.0], dtype=np.float64),
        area_face=mesh.face_areas[: mesh.n_liq + 1],
        vc_face=np.asarray(vc_liq, dtype=np.float64),
        u_face_abs=np.asarray(u_abs_liq, dtype=np.float64),
        u_face_rel=np.asarray(u_abs_liq, dtype=np.float64) - np.asarray(vc_liq, dtype=np.float64),
        current_mass_cell=np.asarray([1.0, 1.0], dtype=np.float64),
        old_mass_cell=np.asarray([1.0, 1.0], dtype=np.float64),
    )
    phase_gas = PhaseVelocityRecovery(
        G_face_abs=np.asarray([0.0, 0.0], dtype=np.float64),
        rho_face=np.asarray([1.0, 1.0], dtype=np.float64),
        area_face=mesh.face_areas[mesh.interface_face_index :],
        vc_face=np.asarray([0.0, 0.0], dtype=np.float64),
        u_face_abs=np.asarray([0.0, 0.0], dtype=np.float64),
        u_face_rel=np.asarray([0.0, 0.0], dtype=np.float64),
        current_mass_cell=np.asarray([1.0], dtype=np.float64),
        old_mass_cell=np.asarray([1.0], dtype=np.float64),
    )
    return VelocityRecoveryPackage(
        liquid=phase_liq,
        gas=phase_gas,
        u_l_if_abs=float(u_abs_liq[-1]),
        u_g_if_abs=0.0,
        G_g_if_abs=0.0,
        diagnostics={},
    )


def _make_iface_pkg(
    *,
    n_liq_full: int,
    n_gas_full: int,
    J_l_full: np.ndarray | None = None,
    N_l_full: np.ndarray | None = None,
    q_l_s: float = 0.0,
    E_l_s: float | None = None,
) -> InterfaceFacePackage:
    if J_l_full is None:
        J_l_full = np.zeros(n_liq_full, dtype=np.float64)
    if N_l_full is None:
        N_l_full = np.asarray(J_l_full, dtype=np.float64)
    if E_l_s is None:
        E_l_s = float(q_l_s)
    return InterfaceFacePackage(
        r_s=2.0,
        area_s=4.0,
        dr_l_s=0.5,
        dr_g_s=0.5,
        dot_a_frozen=0.0,
        Tl_last=300.0,
        Tg_first=320.0,
        Yl_last_full=np.full(n_liq_full, 1.0 / n_liq_full, dtype=np.float64),
        Yg_first_full=np.full(n_gas_full, 1.0 / n_gas_full, dtype=np.float64),
        Xg_first_full=np.full(n_gas_full, 1.0 / n_gas_full, dtype=np.float64),
        Ts=300.0,
        Ys_l_full=np.full(n_liq_full, 1.0 / n_liq_full, dtype=np.float64),
        Ys_g_full=np.full(n_gas_full, 1.0 / n_gas_full, dtype=np.float64),
        Xs_g_full=np.full(n_gas_full, 1.0 / n_gas_full, dtype=np.float64),
        mpp=0.0,
        rho_s_l=1.0,
        rho_s_g=1.0,
        h_s_l=10.0,
        h_s_g=20.0,
        h_liq_species_s_full=np.linspace(10.0, 10.0 + n_liq_full - 1, n_liq_full, dtype=np.float64),
        h_gas_species_s_full=np.linspace(20.0, 20.0 + n_gas_full - 1, n_gas_full, dtype=np.float64),
        k_s_l=1.0,
        k_s_g=1.0,
        D_s_l_full=np.ones(n_liq_full, dtype=np.float64),
        D_s_g_full=np.ones(n_gas_full, dtype=np.float64),
        dTdr_l_s=0.0,
        dTdr_g_s=0.0,
        dYdr_l_s_full=np.zeros(n_liq_full, dtype=np.float64),
        dXdr_g_s_full=np.zeros(n_gas_full, dtype=np.float64),
        J_l_full=np.asarray(J_l_full, dtype=np.float64),
        J_g_full=np.zeros(n_gas_full, dtype=np.float64),
        Vd0_g_full=np.zeros(n_gas_full, dtype=np.float64),
        Vcd_g=0.0,
        N_l_full=np.asarray(N_l_full, dtype=np.float64),
        N_g_full=np.zeros(n_gas_full, dtype=np.float64),
        q_l_s=float(q_l_s),
        q_g_s=0.0,
        q_l_cond=float(q_l_s),
        q_g_cond=0.0,
        q_l_species_diff=0.0,
        q_g_species_diff=0.0,
        E_l_s=float(E_l_s),
        E_g_s=0.0,
        G_g_if_abs=0.0,
        Yeq_g_cond_full=np.full(n_gas_full, 1.0 / n_gas_full, dtype=np.float64),
        gamma_cond_full=np.ones(n_liq_full, dtype=np.float64),
        eps_mass_gas_kinematic=0.0,
    )


def _make_props(*, rho_l: np.ndarray | None = None, h_l: np.ndarray | None = None) -> Any:
    return SimpleNamespace(
        rho_l=np.asarray([1.0, 1.0] if rho_l is None else rho_l, dtype=np.float64),
        h_l=np.asarray([10.0, 10.0] if h_l is None else h_l, dtype=np.float64),
    )


def test_single_component_liquid_energy_only_and_zero_residual() -> None:
    mesh = _make_mesh()
    species_maps = _make_species_maps_single()
    layout = _make_layout_single(mesh)
    state = _make_state(mesh, Yl_full=np.asarray([[1.0], [1.0]]), Yg_full=np.asarray([[0.5, 0.5]]))
    old_star = _make_old_state_current_geom(mesh, state, enthalpy_l=np.asarray([10.0, 10.0]), species_mass_l=np.asarray([[1.0], [1.0]]))
    vel_pkg = _make_velocity_pkg(mesh, u_abs_liq=np.asarray([0.0, 0.0, 0.0]), vc_liq=np.asarray([0.0, 0.0, 0.0]))
    iface_pkg = _make_iface_pkg(n_liq_full=1, n_gas_full=2)

    result = assemble_liquid_residual(
        state_trial=state,
        old_state_current_geom=old_star,
        props_trial=_make_props(),
        mesh=mesh,
        layout=layout,
        species_maps=species_maps,
        liquid_thermo=DummyLiquidThermo(n_species=1),
        liquid_velocity=vel_pkg,
        iface_face_pkg=iface_pkg,
    )

    assert result.energy_rows_global.shape == (2,)
    assert result.species_rows_global.size == 0
    assert np.allclose(result.energy_values, 0.0)


def test_multicomponent_uniform_state_gives_zero_liquid_residual() -> None:
    mesh = _make_mesh()
    species_maps = _make_species_maps_multi()
    layout = _make_layout_multi(mesh)
    state = _make_state(
        mesh,
        Yl_full=np.asarray([[0.7, 0.3], [0.7, 0.3]]),
        Yg_full=np.asarray([[0.3, 0.2, 0.5]]),
    )
    old_star = _make_old_state_current_geom(
        mesh,
        state,
        enthalpy_l=np.asarray([10.0, 10.0]),
        species_mass_l=np.asarray([[0.7, 0.3], [0.7, 0.3]]),
    )
    vel_pkg = _make_velocity_pkg(mesh, u_abs_liq=np.asarray([0.0, 0.0, 0.0]), vc_liq=np.asarray([0.0, 0.0, 0.0]))
    iface_pkg = _make_iface_pkg(n_liq_full=2, n_gas_full=3)

    result = assemble_liquid_residual(
        state_trial=state,
        old_state_current_geom=old_star,
        props_trial=_make_props(),
        mesh=mesh,
        layout=layout,
        species_maps=species_maps,
        liquid_thermo=DummyLiquidThermo(n_species=2),
        liquid_velocity=vel_pkg,
        iface_face_pkg=iface_pkg,
    )

    assert result.species_rows_global.size == mesh.n_liq * species_maps.n_liq_red
    assert np.allclose(result.energy_values, 0.0)
    assert np.allclose(result.species_values, 0.0)


def test_interface_face_fluxes_are_taken_from_interface_package() -> None:
    mesh = _make_mesh()
    species_maps = _make_species_maps_multi()
    layout = _make_layout_multi(mesh)
    state = _make_state(
        mesh,
        Yl_full=np.asarray([[0.7, 0.3], [0.7, 0.3]]),
        Yg_full=np.asarray([[0.3, 0.2, 0.5]]),
    )
    old_star = _make_old_state_current_geom(
        mesh,
        state,
        enthalpy_l=np.asarray([10.0, 10.0]),
        species_mass_l=np.asarray([[0.7, 0.3], [0.7, 0.3]]),
    )
    vel_pkg = _make_velocity_pkg(mesh, u_abs_liq=np.asarray([0.0, 0.0, 0.0]), vc_liq=np.asarray([0.0, 0.0, 0.0]))
    iface_pkg = _make_iface_pkg(
        n_liq_full=2,
        n_gas_full=3,
        J_l_full=np.asarray([0.3, -0.3]),
        N_l_full=np.asarray([0.8, -0.8]),
        q_l_s=2.0,
        E_l_s=5.0,
    )

    result = assemble_liquid_residual(
        state_trial=state,
        old_state_current_geom=old_star,
        props_trial=_make_props(),
        mesh=mesh,
        layout=layout,
        species_maps=species_maps,
        liquid_thermo=DummyLiquidThermo(n_species=2),
        liquid_velocity=vel_pkg,
        iface_face_pkg=iface_pkg,
    )

    assert np.isclose(result.energy_values[-1], 20.0)
    assert np.isclose(result.species_values[-1], 3.2)


def test_convective_term_uses_relative_velocity_not_absolute_velocity() -> None:
    mesh = _make_mesh()
    species_maps = _make_species_maps_single()
    layout = _make_layout_single(mesh)
    state = _make_state(mesh, Yl_full=np.asarray([[1.0], [1.0]]), Yg_full=np.asarray([[0.5, 0.5]]))
    old_star = _make_old_state_current_geom(mesh, state, enthalpy_l=np.asarray([10.0, 20.0]), species_mass_l=np.asarray([[1.0], [1.0]]))
    vel_pkg = _make_velocity_pkg(mesh, u_abs_liq=np.asarray([0.0, 5.0, 5.0]), vc_liq=np.asarray([0.0, 5.0, 5.0]))
    iface_pkg = _make_iface_pkg(n_liq_full=1, n_gas_full=2)

    result = assemble_liquid_residual(
        state_trial=state,
        old_state_current_geom=old_star,
        props_trial=_make_props(h_l=np.asarray([10.0, 20.0])),
        mesh=mesh,
        layout=layout,
        species_maps=species_maps,
        liquid_thermo=DummyLiquidThermo(n_species=1),
        liquid_velocity=vel_pkg,
        iface_face_pkg=iface_pkg,
    )

    assert np.allclose(result.conv_energy, 0.0)
    assert np.allclose(result.energy_values, 0.0)


def test_reduced_full_mapping_keeps_only_reduced_species_rows() -> None:
    mesh = _make_mesh()
    species_maps = _make_species_maps_three_full()
    layout = _make_layout_three_full(mesh)
    state = _make_state(
        mesh,
        Yl_full=np.asarray([[0.2, 0.5, 0.3], [0.2, 0.5, 0.3]]),
        Yg_full=np.asarray([[0.2, 0.2, 0.2, 0.4]]),
    )
    old_star = _make_old_state_current_geom(
        mesh,
        state,
        enthalpy_l=np.asarray([10.0, 10.0]),
        species_mass_l=np.asarray([[0.2, 0.5, 0.3], [0.2, 0.5, 0.3]]),
    )
    vel_pkg = _make_velocity_pkg(mesh, u_abs_liq=np.asarray([0.0, 0.0, 0.0]), vc_liq=np.asarray([0.0, 0.0, 0.0]))
    iface_pkg = _make_iface_pkg(n_liq_full=3, n_gas_full=4, J_l_full=np.asarray([0.1, -0.15, 0.05]), q_l_s=0.0)

    result = assemble_liquid_residual(
        state_trial=state,
        old_state_current_geom=old_star,
        props_trial=_make_props(),
        mesh=mesh,
        layout=layout,
        species_maps=species_maps,
        liquid_thermo=DummyLiquidThermo(n_species=3),
        liquid_velocity=vel_pkg,
        iface_face_pkg=iface_pkg,
    )

    assert result.time_species.shape == (2, 2)
    assert result.species_rows_global.size == 4
    assert np.isclose(result.species_values[2], 0.4)
    assert np.isclose(result.species_values[3], 0.2)


def test_owned_liquid_cells_only_return_owned_rows() -> None:
    mesh = _make_mesh()
    species_maps = _make_species_maps_multi()
    layout = _make_layout_multi(mesh)
    state = _make_state(
        mesh,
        Yl_full=np.asarray([[0.7, 0.3], [0.7, 0.3]]),
        Yg_full=np.asarray([[0.3, 0.2, 0.5]]),
    )
    old_star = _make_old_state_current_geom(
        mesh,
        state,
        enthalpy_l=np.asarray([10.0, 10.0]),
        species_mass_l=np.asarray([[0.7, 0.3], [0.7, 0.3]]),
    )
    vel_pkg = _make_velocity_pkg(mesh, u_abs_liq=np.asarray([0.0, 0.0, 0.0]), vc_liq=np.asarray([0.0, 0.0, 0.0]))
    iface_pkg = _make_iface_pkg(n_liq_full=2, n_gas_full=3)

    result = assemble_liquid_residual(
        state_trial=state,
        old_state_current_geom=old_star,
        props_trial=_make_props(),
        mesh=mesh,
        layout=layout,
        species_maps=species_maps,
        liquid_thermo=DummyLiquidThermo(n_species=2),
        liquid_velocity=vel_pkg,
        iface_face_pkg=iface_pkg,
        owned_liq_cells=np.asarray([1], dtype=np.int64),
    )

    assert result.energy_rows_global.shape == (1,)
    assert result.energy_rows_global[0] == layout.liq_temperature_index(1)
    assert result.species_rows_global.shape == (1,)
    assert result.species_rows_global[0] == layout.liq_species_slice_for_cell(1).start
    assert result.time_energy.shape == (1,)
    assert result.time_species.shape == (1, 1)


def test_velocity_face_arrays_must_match_relative_velocity_contract() -> None:
    mesh = _make_mesh()
    species_maps = _make_species_maps_single()
    layout = _make_layout_single(mesh)
    state = _make_state(mesh, Yl_full=np.asarray([[1.0], [1.0]]), Yg_full=np.asarray([[0.5, 0.5]]))
    old_star = _make_old_state_current_geom(mesh, state, enthalpy_l=np.asarray([10.0, 10.0]), species_mass_l=np.asarray([[1.0], [1.0]]))
    vel_pkg = _make_velocity_pkg(mesh, u_abs_liq=np.asarray([0.0, 0.0, 0.0]), vc_liq=np.asarray([0.0, 0.0, 0.0]))
    vel_pkg = VelocityRecoveryPackage(
        liquid=PhaseVelocityRecovery(
            G_face_abs=vel_pkg.liquid.G_face_abs,
            rho_face=vel_pkg.liquid.rho_face,
            area_face=vel_pkg.liquid.area_face,
            vc_face=vel_pkg.liquid.vc_face,
            u_face_abs=vel_pkg.liquid.u_face_abs,
            u_face_rel=np.asarray([0.0, 1.0, 0.0], dtype=np.float64),
            current_mass_cell=vel_pkg.liquid.current_mass_cell,
            old_mass_cell=vel_pkg.liquid.old_mass_cell,
        ),
        gas=vel_pkg.gas,
        u_l_if_abs=vel_pkg.u_l_if_abs,
        u_g_if_abs=vel_pkg.u_g_if_abs,
        G_g_if_abs=vel_pkg.G_g_if_abs,
        diagnostics=vel_pkg.diagnostics,
    )
    iface_pkg = _make_iface_pkg(n_liq_full=1, n_gas_full=2)

    try:
        assemble_liquid_residual(
            state_trial=state,
            old_state_current_geom=old_star,
            props_trial=_make_props(),
            mesh=mesh,
            layout=layout,
            species_maps=species_maps,
            liquid_thermo=DummyLiquidThermo(n_species=1),
            liquid_velocity=vel_pkg,
            iface_face_pkg=iface_pkg,
        )
    except ValueError as exc:
        assert "u_face_rel" in str(exc)
    else:
        raise AssertionError("assemble_liquid_residual should reject inconsistent liquid face velocities")


def test_internal_face_packages_must_share_explicit_face_ordering(monkeypatch: pytest.MonkeyPatch) -> None:
    mesh = _make_mesh()
    species_maps = _make_species_maps_single()
    layout = _make_layout_single(mesh)
    state = _make_state(mesh, Yl_full=np.asarray([[1.0], [1.0]]), Yg_full=np.asarray([[0.5, 0.5]]))
    old_star = _make_old_state_current_geom(mesh, state, enthalpy_l=np.asarray([10.0, 10.0]), species_mass_l=np.asarray([[1.0], [1.0]]))
    vel_pkg = _make_velocity_pkg(mesh, u_abs_liq=np.asarray([0.0, 0.0, 0.0]), vc_liq=np.asarray([0.0, 0.0, 0.0]))
    iface_pkg = _make_iface_pkg(n_liq_full=1, n_gas_full=2)

    def _bad_convective_pkg(*args, **kwargs) -> ConvectiveInternalFluxPackage:
        return ConvectiveInternalFluxPackage(
            left_cells=np.asarray([0], dtype=np.int64),
            right_cells=np.asarray([1], dtype=np.int64),
            face_indices=np.asarray([99], dtype=np.int64),
            area_face=np.asarray([2.0], dtype=np.float64),
            u_abs_face=np.asarray([0.0], dtype=np.float64),
            vc_face=np.asarray([0.0], dtype=np.float64),
            u_rel_face=np.asarray([0.0], dtype=np.float64),
            upwind_is_left=np.asarray([True]),
            phi_upwind=np.asarray([10.0], dtype=np.float64),
            flux=np.asarray([0.0], dtype=np.float64),
        )

    monkeypatch.setattr(residual_liquid_module, "build_liquid_internal_convective_flux_scalar", _bad_convective_pkg)

    with pytest.raises(ValueError, match="face ordering"):
        assemble_liquid_residual(
            state_trial=state,
            old_state_current_geom=old_star,
            props_trial=_make_props(),
            mesh=mesh,
            layout=layout,
            species_maps=species_maps,
            liquid_thermo=DummyLiquidThermo(n_species=1),
            liquid_velocity=vel_pkg,
            iface_face_pkg=iface_pkg,
        )
