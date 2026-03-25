from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from core.state_recovery import (
    StateRecoveryError,
    _invert_gas_h_to_T,
    _invert_liquid_h_to_T,
    _invert_temperature_monotone_bisection,
    _recover_density,
    _recover_full_mass_fractions,
    _recover_gas_phase_state,
    _recover_liquid_phase_state,
    _recover_specific_enthalpy,
    recover_state_from_contents,
)
from core.types import ConservativeContents, InterfaceState, Mesh1D, RecoveryConfig, RegionSlices, SpeciesMaps


class FakeLinearThermo:
    def __init__(self, cp: float, h_ref: float = 0.0) -> None:
        self.cp = float(cp)
        self.h_ref = float(h_ref)

    def enthalpy_mass(self, T: float, Y_full: np.ndarray) -> float:
        return self.h_ref + self.cp * float(T)


def make_dummy_mesh(n_liq: int, n_gas: int) -> Mesh1D:
    n_cells = n_liq + n_gas
    r_faces = np.linspace(0.0, float(n_cells), n_cells + 1, dtype=np.float64)
    r_centers = 0.5 * (r_faces[:-1] + r_faces[1:])
    volumes = np.ones(n_cells, dtype=np.float64)
    face_areas = np.linspace(0.0, float(n_cells), n_cells + 1, dtype=np.float64)
    dr = np.ones(n_cells, dtype=np.float64)
    return Mesh1D(
        r_faces=r_faces,
        r_centers=r_centers,
        volumes=volumes,
        face_areas=face_areas,
        dr=dr,
        region_slices=RegionSlices(
            liq=slice(0, n_liq),
            gas_near=slice(n_liq, n_cells),
            gas_far=slice(n_cells, n_cells),
            gas_all=slice(n_liq, n_cells),
        ),
        interface_face_index=n_liq,
        interface_cell_liq=n_liq - 1,
        interface_cell_gas=n_liq,
    )


def make_recovery_config() -> RecoveryConfig:
    return RecoveryConfig(
        rho_min=1.0e-12,
        m_min=1.0e-20,
        species_recovery_eps_abs=1.0e-14,
        Y_sum_tol=1.0e-10,
        Y_hard_tol=1.0e-6,
        h_abs_tol=1.0e-12,
        h_rel_tol=1.0e-12,
        h_check_tol=1.0e-8,
        T_step_tol=1.0e-8,
        T_min_l=200.0,
        T_max_l=800.0,
        T_min_g=200.0,
        T_max_g=4000.0,
        liquid_h_inv_max_iter=100,
        cp_min=1.0,
        gas_h_inv_max_iter=100,
        use_cantera_hpy_first=True,
    )


def make_species_maps_single() -> SpeciesMaps:
    return SpeciesMaps(
        liq_full_names=("ethanol",),
        liq_active_names=(),
        liq_closure_name=None,
        gas_full_names=("N2", "O2", "C2H5OH"),
        gas_active_names=("O2", "C2H5OH"),
        gas_closure_name="N2",
        liq_full_to_reduced=np.array([-1], dtype=np.int64),
        liq_reduced_to_full=np.array([], dtype=np.int64),
        gas_full_to_reduced=np.array([-1, 0, 1], dtype=np.int64),
        gas_reduced_to_full=np.array([1, 2], dtype=np.int64),
        liq_full_to_gas_full=np.array([2], dtype=np.int64),
    )


def make_species_maps_multi() -> SpeciesMaps:
    return SpeciesMaps(
        liq_full_names=("A", "B", "C"),
        liq_active_names=("A", "B"),
        liq_closure_name="C",
        gas_full_names=("N2", "A_g", "B_g", "O2"),
        gas_active_names=("A_g", "B_g", "O2"),
        gas_closure_name="N2",
        liq_full_to_reduced=np.array([0, 1, -1], dtype=np.int64),
        liq_reduced_to_full=np.array([0, 1], dtype=np.int64),
        gas_full_to_reduced=np.array([-1, 0, 1, 2], dtype=np.int64),
        gas_reduced_to_full=np.array([1, 2, 3], dtype=np.int64),
        liq_full_to_gas_full=np.array([1, 2, 1], dtype=np.int64),
    )


def make_contents_single(mesh: Mesh1D, *, cp_l: float = 2.0, cp_g: float = 4.0) -> ConservativeContents:
    n_liq = mesh.n_liq
    n_gas = mesh.n_gas
    rho_l = np.array([700.0, 710.0], dtype=np.float64)[:n_liq]
    rho_g = np.array([1.0, 1.1, 1.2], dtype=np.float64)[:n_gas]
    Tl = np.array([300.0, 320.0], dtype=np.float64)[:n_liq]
    Tg = np.array([900.0, 920.0, 940.0], dtype=np.float64)[:n_gas]
    Yl = np.ones((n_liq, 1), dtype=np.float64)
    Yg = np.array(
        [
            [0.70, 0.20, 0.10],
            [0.65, 0.20, 0.15],
            [0.60, 0.25, 0.15],
        ],
        dtype=np.float64,
    )[:n_gas, :]
    liq_vol = mesh.volumes[mesh.region_slices.liq]
    gas_vol = mesh.volumes[mesh.region_slices.gas_all]
    mass_l = rho_l * liq_vol
    species_mass_l = mass_l[:, None] * Yl
    enthalpy_l = mass_l * (cp_l * Tl)
    mass_g = rho_g * gas_vol
    species_mass_g = mass_g[:, None] * Yg
    enthalpy_g = mass_g * (cp_g * Tg)
    return ConservativeContents(
        mass_l=mass_l,
        species_mass_l=species_mass_l,
        enthalpy_l=enthalpy_l,
        mass_g=mass_g,
        species_mass_g=species_mass_g,
        enthalpy_g=enthalpy_g,
    )


def make_contents_multi(mesh: Mesh1D, *, cp_l: float = 2.5, cp_g: float = 4.5) -> ConservativeContents:
    n_liq = mesh.n_liq
    n_gas = mesh.n_gas
    rho_l = np.array([750.0, 760.0], dtype=np.float64)[:n_liq]
    rho_g = np.array([1.2, 1.3], dtype=np.float64)[:n_gas]
    Tl = np.array([330.0, 350.0], dtype=np.float64)[:n_liq]
    Tg = np.array([850.0, 870.0], dtype=np.float64)[:n_gas]
    Yl = np.array([[0.20, 0.30, 0.50], [0.10, 0.20, 0.70]], dtype=np.float64)[:n_liq, :]
    Yg = np.array([[0.60, 0.15, 0.05, 0.20], [0.55, 0.15, 0.10, 0.20]], dtype=np.float64)[:n_gas, :]
    liq_vol = mesh.volumes[mesh.region_slices.liq]
    gas_vol = mesh.volumes[mesh.region_slices.gas_all]
    mass_l = rho_l * liq_vol
    species_mass_l = mass_l[:, None] * Yl
    enthalpy_l = mass_l * (cp_l * Tl)
    mass_g = rho_g * gas_vol
    species_mass_g = mass_g[:, None] * Yg
    enthalpy_g = mass_g * (cp_g * Tg)
    return ConservativeContents(
        mass_l=mass_l,
        species_mass_l=species_mass_l,
        enthalpy_l=enthalpy_l,
        mass_g=mass_g,
        species_mass_g=species_mass_g,
        enthalpy_g=enthalpy_g,
    )


def make_interface_seed(species_maps: SpeciesMaps) -> InterfaceState:
    return InterfaceState(
        Ts=345.0,
        mpp=0.1,
        Ys_g_full=np.full(species_maps.n_gas_full, 1.0 / species_maps.n_gas_full, dtype=np.float64),
        Ys_l_full=np.full(species_maps.n_liq_full, 1.0 / species_maps.n_liq_full, dtype=np.float64),
    )


def test_recover_density_liquid() -> None:
    mass = np.array([2.0, 6.0], dtype=np.float64)
    volumes = np.array([1.0, 2.0], dtype=np.float64)
    assert np.allclose(_recover_density(mass, volumes), np.array([2.0, 3.0]))


def test_recover_density_gas() -> None:
    mass = np.array([1.0, 1.5], dtype=np.float64)
    volumes = np.array([0.5, 0.5], dtype=np.float64)
    assert np.allclose(_recover_density(mass, volumes), np.array([2.0, 3.0]))


def test_recover_specific_enthalpy() -> None:
    enthalpy = np.array([10.0, 18.0], dtype=np.float64)
    mass = np.array([2.0, 3.0], dtype=np.float64)
    assert np.allclose(_recover_specific_enthalpy(enthalpy, mass), np.array([5.0, 6.0]))


def test_recover_full_mass_fractions_multicomponent() -> None:
    mass = np.array([2.0, 4.0], dtype=np.float64)
    species_mass = np.array([[0.5, 1.5], [1.0, 3.0]], dtype=np.float64)
    Y = _recover_full_mass_fractions(species_mass, mass, n_full=2, species_recovery_eps_abs=1.0e-14)
    assert np.allclose(Y, np.array([[0.25, 0.75], [0.25, 0.75]]))


def test_recover_full_mass_fractions_single_component_liquid() -> None:
    mass = np.array([2.0, 4.0], dtype=np.float64)
    species_mass = np.array([[2.0], [4.0]], dtype=np.float64)
    Y = _recover_full_mass_fractions(
        species_mass, mass, n_full=1, single_component_name="ethanol",
        species_recovery_eps_abs=1.0e-14,
    )
    assert np.allclose(Y, np.ones((2, 1)))


def test_invert_liquid_h_to_T_with_linear_thermo() -> None:
    thermo = FakeLinearThermo(cp=2.0)
    cfg = make_recovery_config()
    h = np.array([600.0, 640.0], dtype=np.float64)
    Y = np.ones((2, 1), dtype=np.float64)
    T = _invert_liquid_h_to_T(h, Y, cfg, thermo)
    assert np.allclose(T, np.array([300.0, 320.0]), atol=1.0e-9)


def test_invert_gas_h_to_T_with_linear_thermo() -> None:
    thermo = FakeLinearThermo(cp=4.0)
    cfg = make_recovery_config()
    h = np.array([3600.0, 3680.0], dtype=np.float64)
    Y = np.array([[0.7, 0.2, 0.1], [0.65, 0.20, 0.15]], dtype=np.float64)
    T = _invert_gas_h_to_T(h, Y, cfg, thermo)
    assert np.allclose(T, np.array([900.0, 920.0]), atol=1.0e-9)


def test_invert_temperature_raises_if_target_h_not_bracketed() -> None:
    thermo = FakeLinearThermo(cp=2.0)
    with pytest.raises(StateRecoveryError, match="bracketed"):
        _invert_temperature_monotone_bisection(
            target_h=10000.0,
            y_full=np.array([1.0], dtype=np.float64),
            thermo=thermo,
            T_low=200.0,
            T_high=800.0,
            tol=1.0e-12,
            max_iter=50,
        )


def test_invert_temperature_raises_on_max_iter_exceeded() -> None:
    thermo = FakeLinearThermo(cp=2.0)
    with pytest.raises(StateRecoveryError, match="max_iter"):
        _invert_temperature_monotone_bisection(
            target_h=601.0,
            y_full=np.array([1.0], dtype=np.float64),
            thermo=thermo,
            T_low=200.0,
            T_high=800.0,
            tol=1.0e-16,
            max_iter=1,
        )


def test_recover_liquid_phase_state() -> None:
    mesh = make_dummy_mesh(n_liq=2, n_gas=2)
    contents = make_contents_multi(mesh)
    species_maps = make_species_maps_multi()
    cfg = make_recovery_config()
    rho_l, Yl_full, hl, Tl = _recover_liquid_phase_state(contents, mesh, species_maps, cfg, FakeLinearThermo(cp=2.5))
    assert np.allclose(rho_l, np.array([750.0, 760.0]))
    assert np.allclose(Yl_full, np.array([[0.20, 0.30, 0.50], [0.10, 0.20, 0.70]]))
    assert np.allclose(hl, np.array([825.0, 875.0]))
    assert np.allclose(Tl, np.array([330.0, 350.0]), atol=1.0e-9)


def test_recover_gas_phase_state() -> None:
    mesh = make_dummy_mesh(n_liq=2, n_gas=2)
    contents = make_contents_multi(mesh)
    species_maps = make_species_maps_multi()
    cfg = make_recovery_config()
    rho_g, Yg_full, hg, Tg = _recover_gas_phase_state(contents, mesh, species_maps, cfg, FakeLinearThermo(cp=4.5))
    assert np.allclose(rho_g, np.array([1.2, 1.3]))
    assert np.allclose(Yg_full, np.array([[0.60, 0.15, 0.05, 0.20], [0.55, 0.15, 0.10, 0.20]]))
    assert np.allclose(hg, np.array([3825.0, 3915.0]))
    assert np.allclose(Tg, np.array([850.0, 870.0]), atol=1.0e-9)


def test_recover_state_from_contents_returns_state() -> None:
    mesh = make_dummy_mesh(n_liq=2, n_gas=3)
    contents = make_contents_single(mesh)
    species_maps = make_species_maps_single()
    interface_seed = make_interface_seed(species_maps)
    cfg = make_recovery_config()

    state = recover_state_from_contents(
        contents=contents,
        mesh=mesh,
        species_maps=species_maps,
        recovery_cfg=cfg,
        liquid_thermo=FakeLinearThermo(cp=2.0),
        gas_thermo=FakeLinearThermo(cp=4.0),
        interface_seed=interface_seed,
        time=1.0e-6,
        state_id="recovered",
    )

    assert np.allclose(state.Tl, np.array([300.0, 320.0]))
    assert np.allclose(state.Tg, np.array([900.0, 920.0, 940.0]))
    assert np.allclose(state.Yl_full, np.ones((2, 1)))
    assert np.allclose(
        state.Yg_full,
        np.array([[0.70, 0.20, 0.10], [0.65, 0.20, 0.15], [0.60, 0.25, 0.15]], dtype=np.float64),
    )
    assert np.allclose(state.rho_l, np.array([700.0, 710.0]))
    assert np.allclose(state.rho_g, np.array([1.0, 1.1, 1.2]))
    assert state.interface.Ts == pytest.approx(interface_seed.Ts)
    assert state.time == pytest.approx(1.0e-6)
    assert state.state_id == "recovered"
    assert state.Xg_full is None


def test_recover_state_from_contents_copies_interface_seed_safely() -> None:
    mesh = make_dummy_mesh(n_liq=2, n_gas=3)
    contents = make_contents_single(mesh)
    species_maps = make_species_maps_single()
    interface_seed = make_interface_seed(species_maps)
    cfg = make_recovery_config()

    state = recover_state_from_contents(
        contents=contents,
        mesh=mesh,
        species_maps=species_maps,
        recovery_cfg=cfg,
        liquid_thermo=FakeLinearThermo(cp=2.0),
        gas_thermo=FakeLinearThermo(cp=4.0),
        interface_seed=interface_seed,
    )

    assert state.interface is not interface_seed
    assert state.interface.Ys_g_full is not interface_seed.Ys_g_full
    assert state.interface.Ys_l_full is not interface_seed.Ys_l_full
    state.interface.Ys_g_full[0] = 99.0
    assert interface_seed.Ys_g_full[0] != 99.0


def test_recover_density_rejects_nonpositive_mass() -> None:
    with pytest.raises(StateRecoveryError, match="strictly positive"):
        _recover_density(np.array([1.0, 0.0]), np.array([1.0, 1.0]))


def test_recover_mass_fractions_rejects_species_mass_sum_mismatch() -> None:
    with pytest.raises(StateRecoveryError, match="must match mass"):
        _recover_full_mass_fractions(
            np.array([[0.2, 0.2]], dtype=np.float64),
            np.array([1.0], dtype=np.float64),
            n_full=2,
            species_recovery_eps_abs=1.0e-14,
        )


def test_recover_specific_enthalpy_rejects_nonpositive_mass() -> None:
    with pytest.raises(StateRecoveryError, match="strictly positive"):
        _recover_specific_enthalpy(np.array([1.0]), np.array([0.0]))


def test_recover_state_rejects_temperature_out_of_bounds() -> None:
    mesh = make_dummy_mesh(n_liq=2, n_gas=3)
    contents = make_contents_single(mesh)
    species_maps = make_species_maps_single()
    interface_seed = make_interface_seed(species_maps)
    cfg = make_recovery_config()

    with pytest.raises(StateRecoveryError, match="bracketed"):
        recover_state_from_contents(
            contents=contents,
            mesh=mesh,
            species_maps=species_maps,
            recovery_cfg=cfg,
            liquid_thermo=FakeLinearThermo(cp=0.1),
            gas_thermo=FakeLinearThermo(cp=4.0),
            interface_seed=interface_seed,
        )


def test_recovery_does_not_renormalize_species_mass() -> None:
    with pytest.raises(StateRecoveryError, match="must match mass"):
        _recover_full_mass_fractions(
            np.array([[0.8, 0.8]], dtype=np.float64),
            np.array([1.0], dtype=np.float64),
            n_full=2,
            species_recovery_eps_abs=1.0e-14,
        )


def test_recovery_does_not_clip_temperature() -> None:
    thermo = FakeLinearThermo(cp=2.0)
    with pytest.raises(StateRecoveryError, match="bracketed"):
        _invert_temperature_monotone_bisection(
            target_h=100.0,
            y_full=np.array([1.0], dtype=np.float64),
            thermo=thermo,
            T_low=200.0,
            T_high=800.0,
            tol=1.0e-12,
            max_iter=100,
        )

