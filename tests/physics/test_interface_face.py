from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from core.types import InterfaceState, Mesh1D, RegionSlices, SpeciesMaps, State
from physics.interface_face import build_interface_face_package
from properties.equilibrium import InterfaceEquilibriumResult


def _make_mesh() -> Mesh1D:
    return Mesh1D(
        r_faces=np.array([0.0, 0.8, 1.0, 1.4], dtype=np.float64),
        r_centers=np.array([0.4, 0.9, 1.2], dtype=np.float64),
        volumes=np.ones(3, dtype=np.float64),
        face_areas=np.array([0.0, 8.0, 12.0, 16.0], dtype=np.float64),
        dr=np.array([0.8, 0.2, 0.4], dtype=np.float64),
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


def _make_species_maps() -> SpeciesMaps:
    return SpeciesMaps(
        liq_full_names=("ethanol", "water"),
        liq_active_names=("ethanol",),
        liq_closure_name="water",
        gas_full_names=("C2H5OH", "H2O", "N2"),
        gas_active_names=("C2H5OH", "H2O"),
        gas_closure_name="N2",
        liq_full_to_reduced=np.array([0, -1], dtype=np.int64),
        liq_reduced_to_full=np.array([0], dtype=np.int64),
        gas_full_to_reduced=np.array([0, 1, -1], dtype=np.int64),
        gas_reduced_to_full=np.array([0, 1], dtype=np.int64),
        liq_full_to_gas_full=np.array([0, 1], dtype=np.int64),
    )


def _make_run_cfg() -> SimpleNamespace:
    return SimpleNamespace(
        pressure=101325.0,
        species_maps=_make_species_maps(),
    )


def _make_state() -> State:
    return State(
        Tl=np.array([300.0, 320.0], dtype=np.float64),
        Yl_full=np.array([[0.80, 0.20], [0.70, 0.30]], dtype=np.float64),
        Tg=np.array([360.0], dtype=np.float64),
        Yg_full=np.array([[0.30, 0.10, 0.60]], dtype=np.float64),
        interface=InterfaceState(
            Ts=340.0,
            mpp=0.25,
            Ys_g_full=np.array([0.20, 0.15, 0.65], dtype=np.float64),
            Ys_l_full=np.array([0.55, 0.45], dtype=np.float64),
        ),
        rho_l=np.array([850.0, 900.0], dtype=np.float64),
        rho_g=np.array([1.2], dtype=np.float64),
        hl=np.array([10.0, 20.0], dtype=np.float64),
        hg=np.array([30.0], dtype=np.float64),
        time=0.0,
        state_id="iface_stub",
    )


class _StubLiquidProps:
    def density_mass(self, T: float, Y_full: np.ndarray) -> float:
        return 900.0

    def enthalpy_mass(self, T: float, Y_full: np.ndarray) -> float:
        return 1000.0 + 2.0 * float(T)

    def pure_enthalpy_vector(self, T: float) -> np.ndarray:
        return np.array([10.0 + T, 20.0 + T], dtype=np.float64)

    def conductivity(self, T: float, Y_full: np.ndarray) -> float:
        return 0.2

    def diffusivity(self, T: float, Y_full: np.ndarray) -> np.ndarray:
        return np.array([2.0e-9, 5.0e-9], dtype=np.float64)


class _StubGasProps:
    def __init__(self) -> None:
        self.molecular_weights = np.array([0.046, 0.018, 0.028], dtype=np.float64)

    def density_mass(self, T: float, Y_full: np.ndarray, P: float) -> float:
        return 1.5

    def enthalpy_mass(self, T: float, Y_full: np.ndarray, P: float) -> float:
        return 2000.0 + 3.0 * float(T)

    def species_enthalpies_mass(self, T: float) -> np.ndarray:
        return np.array([100.0 + T, 200.0 + T, 300.0 + T], dtype=np.float64)

    def conductivity(self, T: float, Y_full: np.ndarray, P: float) -> float:
        return 0.05

    def diffusivity(self, T: float, Y_full: np.ndarray, P: float) -> np.ndarray:
        return np.array([1.0e-5, 2.0e-5, 3.0e-5], dtype=np.float64)


class _StubEquilibriumModel:
    liquid_cond_indices = np.array([0, 1], dtype=np.int64)


class _StubEqResult:
    def __init__(self) -> None:
        self.Yg_eq_full = np.array([0.22, 0.18, 0.60], dtype=np.float64)
        self.gamma_cond = np.array([1.1, 0.9], dtype=np.float64)


def _build_package(monkeypatch: pytest.MonkeyPatch, *, state: State | None = None, mesh: Mesh1D | None = None):
    def _fake_compute_interface_equilibrium(model, *, Ts, P, Yl_if_full):
        return _StubEqResult()

    monkeypatch.setattr("physics.interface_face.compute_interface_equilibrium", _fake_compute_interface_equilibrium)
    return build_interface_face_package(
        run_cfg=_make_run_cfg(),
        mesh=_make_mesh() if mesh is None else mesh,
        state=_make_state() if state is None else state,
        gas_props=_StubGasProps(),
        liquid_props=_StubLiquidProps(),
        equilibrium_model=_StubEquilibriumModel(),
        dot_a_frozen=0.4,
    )


def test_interface_geometry_matches_frozen_outer_geometry(monkeypatch: pytest.MonkeyPatch) -> None:
    pkg = _build_package(monkeypatch)
    assert pkg.r_s == pytest.approx(1.0)
    assert pkg.area_s == pytest.approx(12.0)
    assert pkg.dot_a_frozen == pytest.approx(0.4)


def test_negative_dot_a_frozen_is_allowed(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fake_compute_interface_equilibrium(model, *, Ts, P, Yl_if_full):
        return _StubEqResult()

    monkeypatch.setattr("physics.interface_face.compute_interface_equilibrium", _fake_compute_interface_equilibrium)
    pkg = build_interface_face_package(
        run_cfg=_make_run_cfg(),
        mesh=_make_mesh(),
        state=_make_state(),
        gas_props=_StubGasProps(),
        liquid_props=_StubLiquidProps(),
        equilibrium_model=_StubEquilibriumModel(),
        dot_a_frozen=-0.4,
    )
    assert pkg.dot_a_frozen == pytest.approx(-0.4)


def test_interface_distances_are_one_sided_neighbor_distances(monkeypatch: pytest.MonkeyPatch) -> None:
    pkg = _build_package(monkeypatch)
    assert pkg.dr_l_s == pytest.approx(1.0 - 0.9)
    assert pkg.dr_g_s == pytest.approx(1.2 - 1.0)


def test_interface_liquid_temperature_gradient_is_one_sided(monkeypatch: pytest.MonkeyPatch) -> None:
    pkg = _build_package(monkeypatch)
    assert pkg.dTdr_l_s == pytest.approx((340.0 - 320.0) / 0.1)


def test_interface_gas_gradient_uses_mole_fraction_not_mass_fraction(monkeypatch: pytest.MonkeyPatch) -> None:
    pkg = _build_package(monkeypatch)
    Y_first = np.array([0.30, 0.10, 0.60], dtype=np.float64)
    Ys = np.array([0.20, 0.15, 0.65], dtype=np.float64)
    mw = np.array([0.046, 0.018, 0.028], dtype=np.float64)
    X_first = (Y_first / mw) / np.sum(Y_first / mw)
    Xs = (Ys / mw) / np.sum(Ys / mw)
    expected = (X_first - Xs) / 0.2
    dY = (Y_first - Ys) / 0.2
    assert np.allclose(pkg.dXdr_g_s_full, expected)
    assert not np.allclose(pkg.dXdr_g_s_full, dY)


def test_interface_gas_diffusion_uses_mixture_averaged_plus_correction(monkeypatch: pytest.MonkeyPatch) -> None:
    pkg = _build_package(monkeypatch)
    Xs = pkg.Xs_g_full
    dX = pkg.dXdr_g_s_full
    D = pkg.D_s_g_full
    Ys = pkg.Ys_g_full
    expected_vd0 = -(D / np.maximum(Xs, 1.0e-14)) * dX
    expected_vcd = -float(np.sum(Ys * expected_vd0))
    expected_J = -pkg.rho_s_g * Ys * (expected_vd0 + expected_vcd)
    assert np.allclose(pkg.Vd0_g_full, expected_vd0)
    assert pkg.Vcd_g == pytest.approx(expected_vcd)
    assert np.allclose(pkg.J_g_full, expected_J)


def test_interface_liquid_diffusion_respects_closure_zero_sum(monkeypatch: pytest.MonkeyPatch) -> None:
    pkg = _build_package(monkeypatch)
    assert np.isclose(np.sum(pkg.J_l_full), 0.0, atol=1.0e-12, rtol=0.0)


def test_interface_gas_diffusion_zero_sum(monkeypatch: pytest.MonkeyPatch) -> None:
    pkg = _build_package(monkeypatch)
    assert np.isclose(np.sum(pkg.J_g_full), 0.0, atol=1.0e-10, rtol=0.0)


def test_total_species_flux_matches_minus_mppY_plus_J(monkeypatch: pytest.MonkeyPatch) -> None:
    pkg = _build_package(monkeypatch)
    assert np.allclose(pkg.N_l_full, -pkg.mpp * pkg.Ys_l_full + pkg.J_l_full)
    assert np.allclose(pkg.N_g_full, -pkg.mpp * pkg.Ys_g_full + pkg.J_g_full)


def test_interface_heat_flux_keeps_species_diffusion_enthalpy_term(monkeypatch: pytest.MonkeyPatch) -> None:
    state = _make_state()
    state.interface.Ts = state.Tg[0]
    state.Tl[-1] = state.interface.Ts
    pkg = _build_package(monkeypatch, state=state)
    assert pkg.q_l_cond == pytest.approx(0.0)
    assert pkg.q_g_cond == pytest.approx(0.0)
    assert pkg.q_l_species_diff != pytest.approx(0.0)
    assert pkg.q_g_species_diff != pytest.approx(0.0)
    assert pkg.q_l_s == pytest.approx(pkg.q_l_species_diff)
    assert pkg.q_g_s == pytest.approx(pkg.q_g_species_diff)


def test_interface_total_energy_flux_matches_minus_mpph_plus_q(monkeypatch: pytest.MonkeyPatch) -> None:
    pkg = _build_package(monkeypatch)
    assert pkg.E_l_s == pytest.approx(-pkg.mpp * pkg.h_s_l + pkg.q_l_s)
    assert pkg.E_g_s == pytest.approx(-pkg.mpp * pkg.h_s_g + pkg.q_g_s)


def test_interface_equilibrium_diagnostics_are_exposed_but_not_applied_as_residual(monkeypatch: pytest.MonkeyPatch) -> None:
    pkg = _build_package(monkeypatch)
    assert np.allclose(pkg.Yeq_g_cond_full, np.array([0.22, 0.18, 0.60], dtype=np.float64))
    assert np.allclose(pkg.gamma_cond_full, np.array([1.1, 0.9], dtype=np.float64))
    assert not np.allclose(pkg.Ys_g_full, pkg.Yeq_g_cond_full)


def test_gas_side_eq218_strong_imposition_quantity_matches_formula(monkeypatch: pytest.MonkeyPatch) -> None:
    pkg = _build_package(monkeypatch)
    assert pkg.G_g_if_abs == pytest.approx((pkg.rho_s_g * pkg.dot_a_frozen - pkg.mpp) * pkg.area_s)


def test_invalid_mass_fraction_sum_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    state = _make_state()
    state.interface.Ys_g_full = np.array([0.20, 0.15, 0.70], dtype=np.float64)
    with pytest.raises(Exception, match="sum to 1"):
        _build_package(monkeypatch, state=state)


def test_nonpositive_interface_distances_raise(monkeypatch: pytest.MonkeyPatch) -> None:
    mesh = _make_mesh()
    mesh.r_centers[mesh.interface_cell_liq] = mesh.r_faces[mesh.interface_face_index]
    with pytest.raises(Exception, match="dr_l_s"):
        _build_package(monkeypatch, mesh=mesh)


def test_trace_species_in_Xs_g_does_not_create_nan(monkeypatch: pytest.MonkeyPatch) -> None:
    state = _make_state()
    state.interface.Ys_g_full = np.array([1.0e-20, 0.35, 0.65], dtype=np.float64)
    pkg = _build_package(monkeypatch, state=state)
    assert np.all(np.isfinite(pkg.Vd0_g_full))
    assert np.all(np.isfinite(pkg.J_g_full))


def test_multicomponent_liquid_requires_diffusivity(monkeypatch: pytest.MonkeyPatch) -> None:
    class _NoDiffLiquidProps(_StubLiquidProps):
        def diffusivity(self, T: float, Y_full: np.ndarray):
            return None

    def _fake_compute_interface_equilibrium(model, *, Ts, P, Yl_if_full):
        return _StubEqResult()

    monkeypatch.setattr("physics.interface_face.compute_interface_equilibrium", _fake_compute_interface_equilibrium)
    with pytest.raises(Exception, match="requires interface diffusivity"):
        build_interface_face_package(
            run_cfg=_make_run_cfg(),
            mesh=_make_mesh(),
            state=_make_state(),
            gas_props=_StubGasProps(),
            liquid_props=_NoDiffLiquidProps(),
            equilibrium_model=_StubEquilibriumModel(),
            dot_a_frozen=0.4,
        )


def test_interface_face_equilibrium_diagnostics_match_current_equilibrium_api(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fake_compute_interface_equilibrium(model, *, Ts, P, Yl_if_full):
        return InterfaceEquilibriumResult(
            Ts=float(Ts),
            P=float(P),
            Xl_if=np.array([0.45, 0.55], dtype=np.float64),
            Yl_if_full=np.asarray(Yl_if_full, dtype=np.float64).copy(),
            Wl_if=0.025,
            gamma_cond=np.array([1.1, 0.9], dtype=np.float64),
            psat_cond=np.array([2.0e4, 1.0e4], dtype=np.float64),
            latent_cond=np.array([8.0e5, 2.2e6], dtype=np.float64),
            Xg_eq_full=np.array([0.20, 0.15, 0.65], dtype=np.float64),
            Yg_eq_full=np.array([0.22, 0.18, 0.60], dtype=np.float64),
            Wg_eq=0.028,
            condensable_gas_indices=np.array([0, 1], dtype=np.int64),
            diagnostics={"activity_model": "ideal"},
        )

    monkeypatch.setattr("physics.interface_face.compute_interface_equilibrium", _fake_compute_interface_equilibrium)
    pkg = build_interface_face_package(
        run_cfg=_make_run_cfg(),
        mesh=_make_mesh(),
        state=_make_state(),
        gas_props=_StubGasProps(),
        liquid_props=_StubLiquidProps(),
        equilibrium_model=_StubEquilibriumModel(),
        dot_a_frozen=0.4,
    )
    assert np.allclose(pkg.Yeq_g_cond_full, np.array([0.22, 0.18, 0.60], dtype=np.float64))
    assert np.allclose(pkg.gamma_cond_full, np.array([1.1, 0.9], dtype=np.float64))
