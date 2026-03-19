from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

import assembly.residual_global as residual_global_module
from assembly.residual_gas import GasFarFieldBC
from assembly.residual_global import (
    ResidualOwnership,
    assemble_global_residual,
    assemble_global_residual_from_trial_view,
)
from core.layout import UnknownLayout
from core.state_pack import pack_state_to_array
from core.types import (
    ConservativeContents,
    ControlSurfaceMetrics,
    GeometryState,
    InterfaceState,
    Mesh1D,
    OldStateOnCurrentGeometry,
    RegionSlices,
    SpeciesMaps,
    State,
)
from physics.interface_face import InterfaceFacePackage


class DummyLiquidThermo:
    def __init__(self, n_species: int) -> None:
        self.n_species = n_species

    def density_mass_batch(self, T: np.ndarray, Y: np.ndarray) -> np.ndarray:
        return np.ones(T.shape[0], dtype=np.float64)

    def cp_mass_batch(self, T: np.ndarray, Y: np.ndarray) -> np.ndarray:
        return np.ones(T.shape[0], dtype=np.float64)

    def enthalpy_mass_batch(self, T: np.ndarray, Y: np.ndarray) -> np.ndarray:
        return np.asarray(T, dtype=np.float64)

    def conductivity(self, T: float, Y: np.ndarray) -> float:
        return 1.0

    def viscosity(self, T: float, Y: np.ndarray) -> float:
        return 1.0

    def diffusivity(self, T: float, Y: np.ndarray) -> np.ndarray | None:
        if self.n_species == 1:
            return None
        return np.full(self.n_species, 1.0e-9, dtype=np.float64)

    def density_mass(self, T: float, Y: np.ndarray) -> float:
        return 1.0

    def enthalpy_mass(self, T: float, Y: np.ndarray) -> float:
        return float(T)

    def pure_enthalpy_vector(self, T: float) -> np.ndarray:
        return np.full(self.n_species, float(T), dtype=np.float64)


class DummyGasThermo:
    def __init__(self, molecular_weights: np.ndarray) -> None:
        self.molecular_weights = np.asarray(molecular_weights, dtype=np.float64)
        self.n_species = self.molecular_weights.shape[0]
        self.reference_pressure = 101325.0

    def density_mass_batch(self, T: np.ndarray, Y: np.ndarray, p: np.ndarray) -> np.ndarray:
        _ = Y
        _ = p
        return np.ones(T.shape[0], dtype=np.float64)

    def cp_mass_batch(self, T: np.ndarray, Y: np.ndarray, p: np.ndarray) -> np.ndarray:
        _ = T
        _ = Y
        _ = p
        return np.ones(Y.shape[0], dtype=np.float64)

    def enthalpy_mass_batch(self, T: np.ndarray, Y: np.ndarray, p: np.ndarray) -> np.ndarray:
        _ = Y
        _ = p
        return np.asarray(T, dtype=np.float64)

    def conductivity(self, T: float, Y: np.ndarray, p: float) -> float:
        return 1.0

    def viscosity(self, T: float, Y: np.ndarray, p: float) -> float:
        return 1.0

    def diffusivity(self, T: float, Y: np.ndarray, p: float) -> np.ndarray:
        _ = T
        _ = Y
        _ = p
        return np.full(self.n_species, 1.0e-5, dtype=np.float64)

    def density_mass(self, T: float, Y: np.ndarray, p: float) -> float:
        return 1.0

    def enthalpy_mass(self, T: float, Y: np.ndarray, p: float) -> float:
        return float(T)

    def species_enthalpies_mass(self, T: float) -> np.ndarray:
        return np.full(self.n_species, float(T), dtype=np.float64)


def _make_mesh() -> Mesh1D:
    return Mesh1D(
        r_faces=np.asarray([0.0, 1.0, 2.0, 3.0], dtype=np.float64),
        r_centers=np.asarray([0.5, 1.5, 2.5], dtype=np.float64),
        volumes=np.asarray([1.0, 1.0, 1.0], dtype=np.float64),
        face_areas=np.asarray([0.0, 2.0, 4.0, 6.0], dtype=np.float64),
        dr=np.asarray([1.0, 1.0, 1.0], dtype=np.float64),
        region_slices=RegionSlices(
            liq=slice(0, 1),
            gas_near=slice(1, 3),
            gas_far=slice(3, 3),
            gas_all=slice(1, 3),
        ),
        interface_face_index=1,
        interface_cell_liq=0,
        interface_cell_gas=1,
    )


def _make_control_metrics(mesh: Mesh1D) -> ControlSurfaceMetrics:
    return ControlSurfaceMetrics(
        v_c_faces=np.zeros(mesh.n_faces, dtype=np.float64),
        v_c_cells=np.zeros(mesh.n_cells, dtype=np.float64),
        interface_face_index=mesh.interface_face_index,
        region2_outer_face_index=mesh.n_faces - 2,
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
        liq_block=slice(0, 1),
        if_block=slice(1, 4),
        gas_block=slice(4, 8),
        total_size=8,
        liq_temperature_slice=slice(0, 1, 1),
        liq_species_local_slice=slice(1, 1),
        if_temperature_slice=slice(1, 2),
        if_gas_species_slice=slice(2, 3),
        if_mpp_slice=slice(3, 4),
        if_liq_species_slice=slice(4, 4),
        gas_temperature_slice=slice(4, 8, 2),
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
        liq_block=slice(0, 2),
        if_block=slice(2, 7),
        gas_block=slice(7, 13),
        total_size=13,
        liq_temperature_slice=slice(0, 2, 2),
        liq_species_local_slice=slice(1, 2),
        if_temperature_slice=slice(2, 3),
        if_gas_species_slice=slice(3, 5),
        if_mpp_slice=slice(5, 6),
        if_liq_species_slice=slice(6, 7),
        gas_temperature_slice=slice(7, 13, 3),
        gas_species_local_slice=slice(1, 3),
    )


def _make_state_single() -> State:
    return State(
        Tl=np.asarray([300.0], dtype=np.float64),
        Yl_full=np.asarray([[1.0]], dtype=np.float64),
        Tg=np.asarray([300.0, 300.0], dtype=np.float64),
        Yg_full=np.asarray([[0.2, 0.8], [0.2, 0.8]], dtype=np.float64),
        interface=InterfaceState(
            Ts=300.0,
            mpp=0.0,
            Ys_g_full=np.asarray([0.2, 0.8], dtype=np.float64),
            Ys_l_full=np.asarray([1.0], dtype=np.float64),
        ),
        time=0.0,
        state_id="single_global",
    )


def _make_state_multi() -> State:
    return State(
        Tl=np.asarray([300.0], dtype=np.float64),
        Yl_full=np.asarray([[0.6, 0.4]], dtype=np.float64),
        Tg=np.asarray([300.0, 300.0], dtype=np.float64),
        Yg_full=np.asarray([[0.2, 0.1, 0.7], [0.2, 0.1, 0.7]], dtype=np.float64),
        interface=InterfaceState(
            Ts=300.0,
            mpp=0.0,
            Ys_g_full=np.asarray([0.2, 0.1, 0.7], dtype=np.float64),
            Ys_l_full=np.asarray([0.6, 0.4], dtype=np.float64),
        ),
        time=0.0,
        state_id="multi_global",
    )


def _make_old_state_current_geom(mesh: Mesh1D, state: State) -> OldStateOnCurrentGeometry:
    rho_l = np.ones(state.n_liq_cells, dtype=np.float64)
    rho_g = np.ones(state.n_gas_cells, dtype=np.float64)
    hl = np.asarray(state.Tl, dtype=np.float64)
    hg = np.asarray(state.Tg, dtype=np.float64)
    contents = ConservativeContents(
        mass_l=rho_l * mesh.volumes[mesh.liq_slice],
        species_mass_l=rho_l[:, None] * state.Yl_full * mesh.volumes[mesh.liq_slice][:, None],
        enthalpy_l=rho_l * hl * mesh.volumes[mesh.liq_slice],
        mass_g=rho_g * mesh.volumes[mesh.gas_slice],
        species_mass_g=rho_g[:, None] * state.Yg_full * mesh.volumes[mesh.gas_slice][:, None],
        enthalpy_g=rho_g * hg * mesh.volumes[mesh.gas_slice],
    )
    geometry = GeometryState(
        t=0.0,
        dt=1.0,
        a=1.0,
        dot_a=0.0,
        r_end=3.0,
        step_index=0,
        outer_iter_index=0,
    )
    return OldStateOnCurrentGeometry(contents=contents, state=state, geometry=geometry, mesh=mesh)


def _make_run_cfg(species_maps: SpeciesMaps) -> SimpleNamespace:
    return SimpleNamespace(pressure=101325.0, species_maps=species_maps)


def _make_farfield_bc_single() -> GasFarFieldBC:
    return GasFarFieldBC(T_inf=300.0, Yg_inf_full=np.asarray([0.2, 0.8], dtype=np.float64), p_inf=101325.0)


def _make_farfield_bc_multi() -> GasFarFieldBC:
    return GasFarFieldBC(T_inf=300.0, Yg_inf_full=np.asarray([0.2, 0.1, 0.7], dtype=np.float64), p_inf=101325.0)


class _EqModelSingle:
    liquid_cond_indices = np.asarray([0], dtype=np.int64)


class _EqModelMulti:
    liquid_cond_indices = np.asarray([0, 1], dtype=np.int64)


class _EqResult:
    def __init__(self, Yg_eq_full: np.ndarray, gamma_cond: np.ndarray) -> None:
        self.Yg_eq_full = np.asarray(Yg_eq_full, dtype=np.float64)
        self.gamma_cond = np.asarray(gamma_cond, dtype=np.float64)


def _make_iface_pkg_single(*, E_l_s: float = 0.0, E_g_s: float = 0.0) -> InterfaceFacePackage:
    return InterfaceFacePackage(
        r_s=1.0,
        area_s=2.0,
        dr_l_s=0.5,
        dr_g_s=0.5,
        dot_a_frozen=0.0,
        Tl_last=300.0,
        Tg_first=300.0,
        Yl_last_full=np.asarray([1.0], dtype=np.float64),
        Yg_first_full=np.asarray([0.2, 0.8], dtype=np.float64),
        Xg_first_full=np.asarray([0.13207547, 0.86792453], dtype=np.float64),
        Ts=300.0,
        Ys_l_full=np.asarray([1.0], dtype=np.float64),
        Ys_g_full=np.asarray([0.2, 0.8], dtype=np.float64),
        Xs_g_full=np.asarray([0.13207547, 0.86792453], dtype=np.float64),
        mpp=0.0,
        rho_s_l=1.0,
        rho_s_g=1.0,
        h_s_l=300.0,
        h_s_g=300.0,
        h_liq_species_s_full=np.asarray([300.0], dtype=np.float64),
        h_gas_species_s_full=np.asarray([300.0, 300.0], dtype=np.float64),
        k_s_l=1.0,
        k_s_g=1.0,
        D_s_l_full=np.asarray([0.0], dtype=np.float64),
        D_s_g_full=np.asarray([1.0e-5, 1.0e-5], dtype=np.float64),
        dTdr_l_s=0.0,
        dTdr_g_s=0.0,
        dYdr_l_s_full=np.asarray([0.0], dtype=np.float64),
        dXdr_g_s_full=np.asarray([0.0, 0.0], dtype=np.float64),
        J_l_full=np.asarray([0.0], dtype=np.float64),
        J_g_full=np.asarray([0.0, 0.0], dtype=np.float64),
        Vd0_g_full=np.asarray([0.0, 0.0], dtype=np.float64),
        Vcd_g=0.0,
        N_l_full=np.asarray([0.0], dtype=np.float64),
        N_g_full=np.asarray([0.0, 0.0], dtype=np.float64),
        q_l_s=0.0,
        q_g_s=0.0,
        q_l_cond=0.0,
        q_g_cond=0.0,
        q_l_species_diff=0.0,
        q_g_species_diff=0.0,
        E_l_s=float(E_l_s),
        E_g_s=float(E_g_s),
        G_g_if_abs=0.0,
        Yeq_g_cond_full=np.asarray([0.2, 0.8], dtype=np.float64),
        gamma_cond_full=np.asarray([1.0], dtype=np.float64),
        eps_mass_gas_kinematic=0.0,
    )


def test_serial_full_ownership_covers_all_global_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fake_eq(model, *, Ts, P, Yl_if_full):
        return _EqResult(np.asarray([0.2, 0.8], dtype=np.float64), np.asarray([1.0], dtype=np.float64))

    monkeypatch.setattr("physics.interface_face.compute_interface_equilibrium", _fake_eq)
    mesh = _make_mesh()
    species_maps = _make_species_maps_single()
    layout = _make_layout_single(mesh)
    state = _make_state_single()
    result = assemble_global_residual(
        state_trial=state,
        old_state_current_geom=_make_old_state_current_geom(mesh, state),
        mesh=mesh,
        layout=layout,
        species_maps=species_maps,
        ownership=ResidualOwnership(
            owned_liq_cells=np.asarray([0], dtype=np.int64),
            owned_gas_cells=np.asarray([0, 1], dtype=np.int64),
            interface_owner_active=True,
        ),
        run_cfg=_make_run_cfg(species_maps),
        liquid_thermo=DummyLiquidThermo(1),
        gas_thermo=DummyGasThermo(np.asarray([46.0e-3, 28.0e-3])),
        equilibrium_model=_EqModelSingle(),
        control_surface_metrics=_make_control_metrics(mesh),
        farfield_bc=_make_farfield_bc_single(),
    )

    assert result.rows_global.size == layout.total_size
    assert result.values.size == layout.total_size
    assert np.array_equal(np.sort(result.rows_global), np.arange(layout.total_size, dtype=np.int64))
    assert result.diagnostics["row_count"]["total"] == layout.total_size


def test_interface_owner_disabled_returns_only_bulk_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fake_eq(model, *, Ts, P, Yl_if_full):
        return _EqResult(np.asarray([0.2, 0.8], dtype=np.float64), np.asarray([1.0], dtype=np.float64))

    monkeypatch.setattr("physics.interface_face.compute_interface_equilibrium", _fake_eq)
    mesh = _make_mesh()
    species_maps = _make_species_maps_single()
    layout = _make_layout_single(mesh)
    state = _make_state_single()
    result = assemble_global_residual(
        state_trial=state,
        old_state_current_geom=_make_old_state_current_geom(mesh, state),
        mesh=mesh,
        layout=layout,
        species_maps=species_maps,
        ownership=ResidualOwnership(
            owned_liq_cells=np.asarray([0], dtype=np.int64),
            owned_gas_cells=np.asarray([0, 1], dtype=np.int64),
            interface_owner_active=False,
        ),
        run_cfg=_make_run_cfg(species_maps),
        liquid_thermo=DummyLiquidThermo(1),
        gas_thermo=DummyGasThermo(np.asarray([46.0e-3, 28.0e-3])),
        equilibrium_model=_EqModelSingle(),
        control_surface_metrics=_make_control_metrics(mesh),
        farfield_bc=_make_farfield_bc_single(),
    )

    assert result.interface.rows_global.size == 0
    assert result.rows_global.size == result.liquid.rows_global.size + result.gas.rows_global.size
    assert np.unique(result.rows_global).size == result.rows_global.size


def test_multicomponent_case_expands_liquid_and_interface_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fake_eq(model, *, Ts, P, Yl_if_full):
        return _EqResult(np.asarray([0.2, 0.1, 0.7], dtype=np.float64), np.asarray([1.0, 1.0], dtype=np.float64))

    monkeypatch.setattr("physics.interface_face.compute_interface_equilibrium", _fake_eq)
    mesh = _make_mesh()
    species_maps = _make_species_maps_multi()
    layout = _make_layout_multi(mesh)
    state = _make_state_multi()
    result = assemble_global_residual(
        state_trial=state,
        old_state_current_geom=_make_old_state_current_geom(mesh, state),
        mesh=mesh,
        layout=layout,
        species_maps=species_maps,
        ownership=ResidualOwnership(
            owned_liq_cells=np.asarray([0], dtype=np.int64),
            owned_gas_cells=np.asarray([0, 1], dtype=np.int64),
            interface_owner_active=True,
        ),
        run_cfg=_make_run_cfg(species_maps),
        liquid_thermo=DummyLiquidThermo(2),
        gas_thermo=DummyGasThermo(np.asarray([46.0e-3, 18.0e-3, 28.0e-3])),
        equilibrium_model=_EqModelMulti(),
        control_surface_metrics=_make_control_metrics(mesh),
        farfield_bc=_make_farfield_bc_multi(),
    )

    assert result.rows_global.size == layout.total_size
    assert result.liquid.species_rows_global.size == layout.n_liq_red * mesh.n_liq
    assert result.interface.liq_species_rows_global.size == layout.n_liq_red
    assert result.gas.species_rows_global.size == layout.n_gas_red * mesh.n_gas
    assert np.unique(result.rows_global).size == result.rows_global.size


def test_shared_interface_package_change_affects_liquid_interface_and_gas(monkeypatch: pytest.MonkeyPatch) -> None:
    mesh = _make_mesh()
    species_maps = _make_species_maps_single()
    layout = _make_layout_single(mesh)
    state = _make_state_single()
    old_star = _make_old_state_current_geom(mesh, state)

    monkeypatch.setattr("physics.interface_face.compute_interface_equilibrium", lambda model, **kwargs: _EqResult(np.asarray([0.2, 0.8], dtype=np.float64), np.asarray([1.0], dtype=np.float64)))

    base_pkg = _make_iface_pkg_single(E_l_s=0.0, E_g_s=0.0)
    changed_pkg = replace(base_pkg, E_l_s=2.0, E_g_s=3.0)

    monkeypatch.setattr(residual_global_module, "build_interface_face_package", lambda **kwargs: base_pkg)
    base_result = assemble_global_residual(
        state_trial=state,
        old_state_current_geom=old_star,
        mesh=mesh,
        layout=layout,
        species_maps=species_maps,
        ownership=ResidualOwnership(
            owned_liq_cells=np.asarray([0], dtype=np.int64),
            owned_gas_cells=np.asarray([0, 1], dtype=np.int64),
            interface_owner_active=True,
        ),
        run_cfg=_make_run_cfg(species_maps),
        liquid_thermo=DummyLiquidThermo(1),
        gas_thermo=DummyGasThermo(np.asarray([46.0e-3, 28.0e-3])),
        equilibrium_model=_EqModelSingle(),
        control_surface_metrics=_make_control_metrics(mesh),
        farfield_bc=_make_farfield_bc_single(),
    )

    monkeypatch.setattr(residual_global_module, "build_interface_face_package", lambda **kwargs: changed_pkg)
    changed_result = assemble_global_residual(
        state_trial=state,
        old_state_current_geom=old_star,
        mesh=mesh,
        layout=layout,
        species_maps=species_maps,
        ownership=ResidualOwnership(
            owned_liq_cells=np.asarray([0], dtype=np.int64),
            owned_gas_cells=np.asarray([0, 1], dtype=np.int64),
            interface_owner_active=True,
        ),
        run_cfg=_make_run_cfg(species_maps),
        liquid_thermo=DummyLiquidThermo(1),
        gas_thermo=DummyGasThermo(np.asarray([46.0e-3, 28.0e-3])),
        equilibrium_model=_EqModelSingle(),
        control_surface_metrics=_make_control_metrics(mesh),
        farfield_bc=_make_farfield_bc_single(),
    )

    assert not np.isclose(base_result.liquid.energy_values[-1], changed_result.liquid.energy_values[-1])
    assert not np.isclose(base_result.interface.Ts_value[0], changed_result.interface.Ts_value[0])
    assert not np.isclose(base_result.gas.energy_values[0], changed_result.gas.energy_values[0])


def test_trial_view_wrapper_matches_direct_assembly(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fake_eq(model, *, Ts, P, Yl_if_full):
        return _EqResult(np.asarray([0.2, 0.8], dtype=np.float64), np.asarray([1.0], dtype=np.float64))

    monkeypatch.setattr("physics.interface_face.compute_interface_equilibrium", _fake_eq)
    mesh = _make_mesh()
    species_maps = _make_species_maps_single()
    layout = _make_layout_single(mesh)
    state = _make_state_single()
    old_star = _make_old_state_current_geom(mesh, state)
    ownership = ResidualOwnership(
        owned_liq_cells=np.asarray([0], dtype=np.int64),
        owned_gas_cells=np.asarray([0, 1], dtype=np.int64),
        interface_owner_active=True,
    )
    kwargs = dict(
        old_state_current_geom=old_star,
        mesh=mesh,
        layout=layout,
        species_maps=species_maps,
        ownership=ownership,
        run_cfg=_make_run_cfg(species_maps),
        liquid_thermo=DummyLiquidThermo(1),
        gas_thermo=DummyGasThermo(np.asarray([46.0e-3, 28.0e-3])),
        equilibrium_model=_EqModelSingle(),
        control_surface_metrics=_make_control_metrics(mesh),
        farfield_bc=_make_farfield_bc_single(),
    )

    direct = assemble_global_residual(state_trial=state, **kwargs)
    wrapped = assemble_global_residual_from_trial_view(
        vec_trial=pack_state_to_array(state, layout, species_maps),
        base_state=state,
        **kwargs,
    )

    assert np.array_equal(direct.rows_global, wrapped.rows_global)
    assert np.allclose(direct.values, wrapped.values)
