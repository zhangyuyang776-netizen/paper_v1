from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest

import assembly.residual_gas as residual_gas_module
from assembly.residual_gas import GasFarFieldBC, assemble_gas_residual
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
from physics.flux_gas import GasFaceDiffusionPackage, GasFaceEnergyPackage


@dataclass
class DummyGasThermo:
    molecular_weights: np.ndarray

    def density_mass(self, T: float, Y: np.ndarray, p: float) -> float:
        _ = Y
        _ = p
        return 1.0

    def enthalpy_mass(self, T: float, Y: np.ndarray, p: float) -> float:
        _ = Y
        _ = p
        return float(T)

    def conductivity(self, T: float, Y: np.ndarray, p: float) -> float:
        _ = T
        _ = Y
        _ = p
        return 1.0

    def diffusivity(self, T: float, Y: np.ndarray, p: float) -> np.ndarray:
        _ = T
        _ = Y
        _ = p
        return np.full(self.molecular_weights.shape[0], 1.0e-5, dtype=np.float64)

    def species_enthalpies_mass(self, T: float) -> np.ndarray:
        return np.full(self.molecular_weights.shape[0], float(T), dtype=np.float64)


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


def _make_species_maps() -> SpeciesMaps:
    return SpeciesMaps(
        liq_full_names=("ethanol",),
        liq_active_names=(),
        liq_closure_name="ethanol",
        gas_full_names=("C2H5OH", "H2O", "N2"),
        gas_active_names=("C2H5OH", "H2O"),
        gas_closure_name="N2",
        liq_full_to_reduced=np.asarray([-1], dtype=np.int64),
        liq_reduced_to_full=np.asarray([], dtype=np.int64),
        gas_full_to_reduced=np.asarray([0, 1, -1], dtype=np.int64),
        gas_reduced_to_full=np.asarray([0, 1], dtype=np.int64),
        liq_full_to_gas_full=np.asarray([0], dtype=np.int64),
    )


def _make_layout(mesh: Mesh1D) -> UnknownLayout:
    return UnknownLayout(
        unknowns_profile="U_A",
        n_liq_cells=mesh.n_liq,
        n_gas_cells=mesh.n_gas,
        n_liq_red=0,
        n_gas_red=2,
        liq_cell_width=1,
        gas_cell_width=3,
        n_if_unknowns=4,
        liq_block=slice(0, 1),
        if_block=slice(1, 5),
        gas_block=slice(5, 11),
        total_size=11,
        liq_temperature_slice=slice(0, 1, 1),
        liq_species_local_slice=slice(1, 1),
        if_temperature_slice=slice(1, 2),
        if_gas_species_slice=slice(2, 4),
        if_mpp_slice=slice(4, 5),
        if_liq_species_slice=slice(5, 5),
        gas_temperature_slice=slice(5, 11, 3),
        gas_species_local_slice=slice(1, 3),
    )


def _make_state(
    *,
    Yg_full: np.ndarray | None = None,
    Tg: np.ndarray | None = None,
    rho_g: np.ndarray | None = None,
    hg: np.ndarray | None = None,
) -> State:
    gas_y = np.asarray(
        [[0.2, 0.3, 0.5], [0.2, 0.3, 0.5]] if Yg_full is None else Yg_full,
        dtype=np.float64,
    )
    gas_T = np.asarray([300.0, 300.0] if Tg is None else Tg, dtype=np.float64)
    gas_rho = np.asarray([1.0, 1.0] if rho_g is None else rho_g, dtype=np.float64)
    gas_h = np.asarray([300.0, 300.0] if hg is None else hg, dtype=np.float64)
    return State(
        Tl=np.asarray([300.0], dtype=np.float64),
        Yl_full=np.asarray([[1.0]], dtype=np.float64),
        Tg=gas_T,
        Yg_full=gas_y,
        interface=InterfaceState(
            Ts=300.0,
            mpp=0.0,
            Ys_g_full=np.asarray(gas_y[0, :], dtype=np.float64),
            Ys_l_full=np.asarray([1.0], dtype=np.float64),
        ),
        rho_l=np.asarray([1.0], dtype=np.float64),
        rho_g=gas_rho,
        hl=np.asarray([100.0], dtype=np.float64),
        hg=gas_h,
    )


def _make_old_state_current_geom(mesh: Mesh1D, state: State) -> OldStateOnCurrentGeometry:
    contents = ConservativeContents(
        mass_l=np.asarray([1.0], dtype=np.float64),
        species_mass_l=np.asarray([[1.0]], dtype=np.float64),
        enthalpy_l=np.asarray([100.0], dtype=np.float64),
        mass_g=np.asarray([1.0, 1.0], dtype=np.float64),
        species_mass_g=np.asarray(state.rho_g[:, None] * state.Yg_full, dtype=np.float64),
        enthalpy_g=np.asarray(state.rho_g * state.hg, dtype=np.float64),
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


def _make_velocity_pkg(
    mesh: Mesh1D,
    *,
    u_abs_gas: np.ndarray,
    vc_gas: np.ndarray,
    G_gas: np.ndarray,
) -> VelocityRecoveryPackage:
    gas_u_abs = np.asarray(u_abs_gas, dtype=np.float64)
    gas_vc = np.asarray(vc_gas, dtype=np.float64)
    gas_G = np.asarray(G_gas, dtype=np.float64)
    phase_liq = PhaseVelocityRecovery(
        G_face_abs=np.asarray([0.0, 0.0], dtype=np.float64),
        rho_face=np.asarray([1.0, 1.0], dtype=np.float64),
        area_face=mesh.face_areas[: mesh.n_liq + 1],
        vc_face=np.asarray([0.0, 0.0], dtype=np.float64),
        u_face_abs=np.asarray([0.0, 0.0], dtype=np.float64),
        u_face_rel=np.asarray([0.0, 0.0], dtype=np.float64),
        current_mass_cell=np.asarray([1.0], dtype=np.float64),
        old_mass_cell=np.asarray([1.0], dtype=np.float64),
    )
    phase_gas = PhaseVelocityRecovery(
        G_face_abs=gas_G,
        rho_face=np.asarray([1.0, 1.0, 1.0], dtype=np.float64),
        area_face=mesh.face_areas[mesh.interface_face_index :],
        vc_face=gas_vc,
        u_face_abs=gas_u_abs,
        u_face_rel=gas_u_abs - gas_vc,
        current_mass_cell=np.asarray([1.0, 1.0], dtype=np.float64),
        old_mass_cell=np.asarray([1.0, 1.0], dtype=np.float64),
    )
    return VelocityRecoveryPackage(
        liquid=phase_liq,
        gas=phase_gas,
        u_l_if_abs=0.0,
        u_g_if_abs=float(gas_u_abs[0]),
        G_g_if_abs=float(gas_G[0]),
        diagnostics={},
    )


def _make_iface_pkg(
    *,
    E_g_s: float = 0.0,
    q_g_s: float = 0.0,
    N_g_full: np.ndarray | None = None,
    J_g_full: np.ndarray | None = None,
    G_g_if_abs: float = 0.0,
) -> InterfaceFacePackage:
    N_g = np.zeros(3, dtype=np.float64) if N_g_full is None else np.asarray(N_g_full, dtype=np.float64)
    J_g = np.zeros(3, dtype=np.float64) if J_g_full is None else np.asarray(J_g_full, dtype=np.float64)
    return InterfaceFacePackage(
        r_s=1.0,
        area_s=2.0,
        dr_l_s=0.5,
        dr_g_s=0.5,
        dot_a_frozen=0.0,
        Tl_last=300.0,
        Tg_first=300.0,
        Yl_last_full=np.asarray([1.0], dtype=np.float64),
        Yg_first_full=np.asarray([0.2, 0.3, 0.5], dtype=np.float64),
        Xg_first_full=np.asarray([0.4, 0.3, 0.3], dtype=np.float64),
        Ts=300.0,
        Ys_l_full=np.asarray([1.0], dtype=np.float64),
        Ys_g_full=np.asarray([0.2, 0.3, 0.5], dtype=np.float64),
        Xs_g_full=np.asarray([0.4, 0.3, 0.3], dtype=np.float64),
        mpp=0.0,
        rho_s_l=1.0,
        rho_s_g=1.0,
        h_s_l=100.0,
        h_s_g=300.0,
        h_liq_species_s_full=np.asarray([100.0], dtype=np.float64),
        h_gas_species_s_full=np.asarray([300.0, 300.0, 300.0], dtype=np.float64),
        k_s_l=1.0,
        k_s_g=1.0,
        D_s_l_full=np.asarray([0.0], dtype=np.float64),
        D_s_g_full=np.asarray([1.0e-5, 1.0e-5, 1.0e-5], dtype=np.float64),
        dTdr_l_s=0.0,
        dTdr_g_s=0.0,
        dYdr_l_s_full=np.asarray([0.0], dtype=np.float64),
        dXdr_g_s_full=np.asarray([0.0, 0.0, 0.0], dtype=np.float64),
        J_l_full=np.asarray([0.0], dtype=np.float64),
        J_g_full=J_g,
        Vd0_g_full=np.asarray([0.0, 0.0, 0.0], dtype=np.float64),
        Vcd_g=0.0,
        N_l_full=np.asarray([0.0], dtype=np.float64),
        N_g_full=N_g,
        q_l_s=0.0,
        q_g_s=float(q_g_s),
        q_l_cond=0.0,
        q_g_cond=float(q_g_s),
        q_l_species_diff=0.0,
        q_g_species_diff=0.0,
        E_l_s=0.0,
        E_g_s=float(E_g_s),
        G_g_if_abs=float(G_g_if_abs),
        Yeq_g_cond_full=np.asarray([0.2, 0.3, 0.5], dtype=np.float64),
        gamma_cond_full=np.asarray([1.0], dtype=np.float64),
        eps_mass_gas_kinematic=0.0,
    )


def _make_props(state: State) -> SimpleNamespace:
    return SimpleNamespace(
        rho_g=np.asarray(state.rho_g, dtype=np.float64),
        h_g=np.asarray(state.hg, dtype=np.float64),
    )


def _make_farfield_bc(*, T_inf: float = 300.0, Y_inf: np.ndarray | None = None) -> GasFarFieldBC:
    return GasFarFieldBC(
        T_inf=float(T_inf),
        Yg_inf_full=np.asarray([0.2, 0.3, 0.5] if Y_inf is None else Y_inf, dtype=np.float64),
        p_inf=101325.0,
    )


def test_uniform_state_with_zero_fluxes_gives_zero_gas_residual() -> None:
    mesh = _make_mesh()
    maps = _make_species_maps()
    layout = _make_layout(mesh)
    state = _make_state()
    old_star = _make_old_state_current_geom(mesh, state)
    vel_pkg = _make_velocity_pkg(
        mesh,
        u_abs_gas=np.asarray([0.0, 0.0, 0.0]),
        vc_gas=np.asarray([0.0, 0.0, 0.0]),
        G_gas=np.asarray([0.0, 0.0, 0.0]),
    )
    iface_pkg = _make_iface_pkg(G_g_if_abs=0.0)
    result = assemble_gas_residual(
        state_trial=state,
        old_state_current_geom=old_star,
        props_trial=_make_props(state),
        mesh=mesh,
        layout=layout,
        species_maps=maps,
        gas_thermo=DummyGasThermo(np.asarray([46.0e-3, 18.0e-3, 28.0e-3])),
        gas_velocity=vel_pkg,
        iface_face_pkg=iface_pkg,
        farfield_bc=_make_farfield_bc(),
    )

    assert result.energy_rows_global.shape == (2,)
    assert result.species_rows_global.shape == (4,)
    assert np.allclose(result.energy_values, 0.0)
    assert np.allclose(result.species_values, 0.0)


def test_interface_face_package_directly_drives_first_gas_cell_fluxes() -> None:
    mesh = _make_mesh()
    maps = _make_species_maps()
    layout = _make_layout(mesh)
    state = _make_state()
    old_star = _make_old_state_current_geom(mesh, state)
    vel_pkg = _make_velocity_pkg(
        mesh,
        u_abs_gas=np.asarray([0.0, 0.0, 0.0]),
        vc_gas=np.asarray([0.0, 0.0, 0.0]),
        G_gas=np.asarray([0.0, 0.0, 0.0]),
    )
    thermo = DummyGasThermo(np.asarray([46.0e-3, 18.0e-3, 28.0e-3]))

    base_result = assemble_gas_residual(
        state_trial=state,
        old_state_current_geom=old_star,
        props_trial=_make_props(state),
        mesh=mesh,
        layout=layout,
        species_maps=maps,
        gas_thermo=thermo,
        gas_velocity=vel_pkg,
        iface_face_pkg=_make_iface_pkg(G_g_if_abs=0.0),
        farfield_bc=_make_farfield_bc(),
    )
    changed_result = assemble_gas_residual(
        state_trial=state,
        old_state_current_geom=old_star,
        props_trial=_make_props(state),
        mesh=mesh,
        layout=layout,
        species_maps=maps,
        gas_thermo=thermo,
        gas_velocity=vel_pkg,
        iface_face_pkg=_make_iface_pkg(E_g_s=3.0, N_g_full=np.asarray([0.4, 0.0, -0.4]), G_g_if_abs=0.0),
        farfield_bc=_make_farfield_bc(),
    )

    assert np.isclose(changed_result.energy_values[0] - base_result.energy_values[0], -6.0)
    assert np.isclose(changed_result.species_values[0] - base_result.species_values[0], -0.8)


def test_gas_interface_mass_flux_must_match_velocity_recovery_start() -> None:
    mesh = _make_mesh()
    maps = _make_species_maps()
    layout = _make_layout(mesh)
    state = _make_state()
    old_star = _make_old_state_current_geom(mesh, state)
    vel_pkg = _make_velocity_pkg(
        mesh,
        u_abs_gas=np.asarray([0.0, 0.0, 0.0]),
        vc_gas=np.asarray([0.0, 0.0, 0.0]),
        G_gas=np.asarray([0.5, 0.0, 0.0]),
    )

    with pytest.raises(ValueError, match="G_g_if_abs"):
        assemble_gas_residual(
            state_trial=state,
            old_state_current_geom=old_star,
            props_trial=_make_props(state),
            mesh=mesh,
            layout=layout,
            species_maps=maps,
            gas_thermo=DummyGasThermo(np.asarray([46.0e-3, 18.0e-3, 28.0e-3])),
            gas_velocity=vel_pkg,
            iface_face_pkg=_make_iface_pkg(G_g_if_abs=0.0),
            farfield_bc=_make_farfield_bc(),
        )


def test_farfield_convective_flux_uses_boundary_state_for_inflow_and_cell_state_for_outflow() -> None:
    mesh = _make_mesh()
    maps = _make_species_maps()
    layout = _make_layout(mesh)
    state = _make_state(Tg=np.asarray([300.0, 300.0]), hg=np.asarray([300.0, 300.0]))
    old_star = _make_old_state_current_geom(mesh, state)
    iface_pkg = _make_iface_pkg(G_g_if_abs=0.0)
    thermo = DummyGasThermo(np.asarray([46.0e-3, 18.0e-3, 28.0e-3]))

    inflow = assemble_gas_residual(
        state_trial=state,
        old_state_current_geom=old_star,
        props_trial=_make_props(state),
        mesh=mesh,
        layout=layout,
        species_maps=maps,
        gas_thermo=thermo,
        gas_velocity=_make_velocity_pkg(
            mesh,
            u_abs_gas=np.asarray([0.0, 0.0, -2.0]),
            vc_gas=np.asarray([0.0, 0.0, 0.0]),
            G_gas=np.asarray([0.0, 0.0, -2.0]),
        ),
        iface_face_pkg=iface_pkg,
        farfield_bc=_make_farfield_bc(T_inf=400.0),
    )
    outflow = assemble_gas_residual(
        state_trial=state,
        old_state_current_geom=old_star,
        props_trial=_make_props(state),
        mesh=mesh,
        layout=layout,
        species_maps=maps,
        gas_thermo=thermo,
        gas_velocity=_make_velocity_pkg(
            mesh,
            u_abs_gas=np.asarray([0.0, 0.0, 2.0]),
            vc_gas=np.asarray([0.0, 0.0, 0.0]),
            G_gas=np.asarray([0.0, 0.0, 2.0]),
        ),
        iface_face_pkg=iface_pkg,
        farfield_bc=_make_farfield_bc(T_inf=400.0),
    )

    assert np.isclose(inflow.conv_energy[-1], -4800.0)
    assert np.isclose(outflow.conv_energy[-1], 3600.0)


def test_owned_gas_cells_only_return_owned_rows() -> None:
    mesh = _make_mesh()
    maps = _make_species_maps()
    layout = _make_layout(mesh)
    state = _make_state()
    old_star = _make_old_state_current_geom(mesh, state)
    result = assemble_gas_residual(
        state_trial=state,
        old_state_current_geom=old_star,
        props_trial=_make_props(state),
        mesh=mesh,
        layout=layout,
        species_maps=maps,
        gas_thermo=DummyGasThermo(np.asarray([46.0e-3, 18.0e-3, 28.0e-3])),
        gas_velocity=_make_velocity_pkg(
            mesh,
            u_abs_gas=np.asarray([0.0, 0.0, 0.0]),
            vc_gas=np.asarray([0.0, 0.0, 0.0]),
            G_gas=np.asarray([0.0, 0.0, 0.0]),
        ),
        iface_face_pkg=_make_iface_pkg(G_g_if_abs=0.0),
        farfield_bc=_make_farfield_bc(),
        owned_gas_cells=np.asarray([1], dtype=np.int64),
    )

    assert result.energy_rows_global.shape == (1,)
    assert result.energy_rows_global[0] == layout.gas_temperature_index(1)
    assert result.species_rows_global.shape == (2,)
    assert np.array_equal(
        result.species_rows_global,
        np.arange(
            layout.gas_species_slice_for_cell(1).start,
            layout.gas_species_slice_for_cell(1).stop,
            dtype=np.int64,
        ),
    )
    assert result.time_species.shape == (2, 1)


def test_internal_face_packages_must_not_include_special_faces(monkeypatch: pytest.MonkeyPatch) -> None:
    mesh = _make_mesh()
    maps = _make_species_maps()
    layout = _make_layout(mesh)
    state = _make_state()
    old_star = _make_old_state_current_geom(mesh, state)
    vel_pkg = _make_velocity_pkg(
        mesh,
        u_abs_gas=np.asarray([0.0, 0.0, 0.0]),
        vc_gas=np.asarray([0.0, 0.0, 0.0]),
        G_gas=np.asarray([0.0, 0.0, 0.0]),
    )

    def _bad_diff_pkg(*args, **kwargs) -> GasFaceDiffusionPackage:
        return GasFaceDiffusionPackage(
            left_cells=np.asarray([0], dtype=np.int64),
            right_cells=np.asarray([1], dtype=np.int64),
            face_indices=np.asarray([mesh.interface_face_index], dtype=np.int64),
            X_face_full=np.asarray([[0.4, 0.3, 0.3]], dtype=np.float64),
            Y_face_full=np.asarray([[0.2, 0.3, 0.5]], dtype=np.float64),
            dXdr_full=np.zeros((1, 3), dtype=np.float64),
            rho_face=np.asarray([1.0], dtype=np.float64),
            D_face_full=np.full((1, 3), 1.0e-5, dtype=np.float64),
            Vd0_face_full=np.zeros((1, 3), dtype=np.float64),
            Vcd_face=np.asarray([0.0], dtype=np.float64),
            J_diff_full=np.zeros((1, 3), dtype=np.float64),
        )

    def _bad_conv_pkg(*args, **kwargs) -> ConvectiveInternalFluxPackage:
        return ConvectiveInternalFluxPackage(
            left_cells=np.asarray([0], dtype=np.int64),
            right_cells=np.asarray([1], dtype=np.int64),
            face_indices=np.asarray([mesh.interface_face_index], dtype=np.int64),
            area_face=np.asarray([2.0], dtype=np.float64),
            u_abs_face=np.asarray([0.0], dtype=np.float64),
            vc_face=np.asarray([0.0], dtype=np.float64),
            u_rel_face=np.asarray([0.0], dtype=np.float64),
            upwind_is_left=np.asarray([True]),
            phi_upwind=np.asarray([0.0], dtype=np.float64),
            flux=np.asarray([0.0], dtype=np.float64),
        )

    def _bad_energy_pkg(*args, **kwargs) -> GasFaceEnergyPackage:
        return GasFaceEnergyPackage(
            left_cells=np.asarray([0], dtype=np.int64),
            right_cells=np.asarray([1], dtype=np.int64),
            face_indices=np.asarray([mesh.interface_face_index], dtype=np.int64),
            grad_T=np.asarray([0.0], dtype=np.float64),
            k_face=np.asarray([1.0], dtype=np.float64),
            h_face_full=np.asarray([[300.0, 300.0, 300.0]], dtype=np.float64),
            q_cond=np.asarray([0.0], dtype=np.float64),
            q_species_diff=np.asarray([0.0], dtype=np.float64),
            q_total=np.asarray([0.0], dtype=np.float64),
        )

    monkeypatch.setattr(residual_gas_module, "build_gas_internal_diffusion_package", _bad_diff_pkg)
    monkeypatch.setattr(residual_gas_module, "build_gas_internal_energy_flux_package", _bad_energy_pkg)
    monkeypatch.setattr(residual_gas_module, "build_gas_internal_convective_flux_scalar", _bad_conv_pkg)
    monkeypatch.setattr(residual_gas_module, "build_gas_internal_convective_flux_vector", _bad_conv_pkg)

    with pytest.raises(ValueError, match="must not include the interface face"):
        assemble_gas_residual(
            state_trial=state,
            old_state_current_geom=old_star,
            props_trial=_make_props(state),
            mesh=mesh,
            layout=layout,
            species_maps=maps,
            gas_thermo=DummyGasThermo(np.asarray([46.0e-3, 18.0e-3, 28.0e-3])),
            gas_velocity=vel_pkg,
            iface_face_pkg=_make_iface_pkg(G_g_if_abs=0.0),
            farfield_bc=_make_farfield_bc(),
        )
