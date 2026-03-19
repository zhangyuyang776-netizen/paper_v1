from __future__ import annotations

import numpy as np

from core.types import InterfaceState, Mesh1D, RegionSlices, State
from physics.flux_gas import (
    build_gas_farfield_boundary_flux_package,
    build_gas_internal_diffusion_package,
    build_gas_internal_energy_flux_package,
)


def _make_stub_mesh() -> Mesh1D:
    return Mesh1D(
        r_faces=np.array([0.0, 1.0, 2.0, 3.0, 4.0], dtype=np.float64),
        r_centers=np.array([0.5, 1.5, 2.5, 3.5], dtype=np.float64),
        volumes=np.ones(4, dtype=np.float64),
        face_areas=np.array([0.0, 1.0, 2.0, 3.0, 4.0], dtype=np.float64),
        dr=np.ones(4, dtype=np.float64),
        region_slices=RegionSlices(
            liq=slice(0, 1),
            gas_near=slice(1, 2),
            gas_far=slice(2, 4),
            gas_all=slice(1, 4),
        ),
        interface_face_index=1,
        interface_cell_liq=0,
        interface_cell_gas=1,
    )


def _make_state(
    *,
    Tg: np.ndarray | None = None,
    Yg_full: np.ndarray | None = None,
    rho_g: np.ndarray | None = None,
) -> State:
    Tg_values = np.array([900.0, 960.0, 1020.0], dtype=np.float64) if Tg is None else np.asarray(Tg, dtype=np.float64)
    Yg_values = (
        np.array(
            [
                [0.20, 0.30, 0.50],
                [0.30, 0.20, 0.50],
                [0.25, 0.25, 0.50],
            ],
            dtype=np.float64,
        )
        if Yg_full is None
        else np.asarray(Yg_full, dtype=np.float64)
    )
    rho_g_values = np.array([1.00, 1.20, 1.40], dtype=np.float64) if rho_g is None else np.asarray(rho_g, dtype=np.float64)
    return State(
        Tl=np.array([298.0], dtype=np.float64),
        Yl_full=np.array([[1.0]], dtype=np.float64),
        Tg=Tg_values,
        Yg_full=Yg_values,
        interface=InterfaceState(
            Ts=298.0,
            mpp=0.0,
            Ys_g_full=np.array([0.12, 0.18, 0.70], dtype=np.float64),
            Ys_l_full=np.array([1.0], dtype=np.float64),
        ),
        rho_l=np.array([800.0], dtype=np.float64),
        rho_g=rho_g_values,
        hl=np.array([0.0], dtype=np.float64),
        hg=np.array([1.0, 2.0, 3.0], dtype=np.float64),
        time=0.0,
        state_id="stub_flux_gas",
    )


def _mass_to_mole_rows(Y_full: np.ndarray, molecular_weights: np.ndarray) -> np.ndarray:
    mole_basis = Y_full / molecular_weights[None, :]
    return mole_basis / np.sum(mole_basis, axis=1, keepdims=True)


class _StubGasProps:
    def __init__(self) -> None:
        self.molecular_weights = np.array([0.020, 0.040, 0.030], dtype=np.float64)
        self.conductivity_calls = 0
        self.diffusivity_calls = 0
        self.species_enthalpy_calls = 0
        self.density_calls = 0

    def density_mass(self, T: float, Y_full: np.ndarray, P: float) -> float:
        self.density_calls += 1
        return 0.8 + 2.0e-4 * float(T) + 1.0e-7 * float(P) + 0.1 * float(Y_full[0])

    def conductivity(self, T: float, Y_full: np.ndarray, P: float) -> float:
        self.conductivity_calls += 1
        return 0.02 + 1.0e-5 * float(T) + 5.0e-4 * float(Y_full[0]) + 1.0e-9 * float(P)

    def diffusivity(self, T: float, Y_full: np.ndarray, P: float) -> np.ndarray:
        self.diffusivity_calls += 1
        return np.array(
            [
                1.0e-5 + 1.0e-8 * float(T),
                1.4e-5 + 1.0e-8 * float(T),
                2.0e-5 + 1.0e-8 * float(T),
            ],
            dtype=np.float64,
        )

    def species_enthalpies_mass(self, T: float) -> np.ndarray:
        self.species_enthalpy_calls += 1
        T_value = float(T)
        return np.array(
            [1000.0 + 1.0 * T_value, 2000.0 + 2.0 * T_value, 3000.0 + 3.0 * T_value],
            dtype=np.float64,
        )


def test_gas_diffusion_uses_mole_fraction_gradient_not_mass_fraction_gradient() -> None:
    mesh = _make_stub_mesh()
    state = _make_state()
    gas_props = _StubGasProps()

    diff_pkg = build_gas_internal_diffusion_package(mesh, state, gas_props, p_env=101325.0)

    X_cell = _mass_to_mole_rows(state.Yg_full, gas_props.molecular_weights)
    expected_dX = X_cell[1:, :] - X_cell[:-1, :]
    dY = state.Yg_full[1:, :] - state.Yg_full[:-1, :]

    assert np.allclose(diff_pkg.dXdr_full, expected_dX)
    assert not np.allclose(diff_pkg.dXdr_full, dY)


def test_gas_diffusion_matches_mixture_averaged_correction_form() -> None:
    mesh = _make_stub_mesh()
    state = _make_state()
    gas_props = _StubGasProps()

    diff_pkg = build_gas_internal_diffusion_package(mesh, state, gas_props, p_env=101325.0)

    X_cell = _mass_to_mole_rows(state.Yg_full, gas_props.molecular_weights)
    X_face = 0.5 * (X_cell[:-1, :] + X_cell[1:, :])
    Y_face = 0.5 * (state.Yg_full[:-1, :] + state.Yg_full[1:, :])
    dXdr = X_cell[1:, :] - X_cell[:-1, :]
    rho_face = 0.5 * (state.rho_g[:-1] + state.rho_g[1:])
    D_cell = np.vstack([gas_props.diffusivity(float(T), state.Yg_full[i, :], 101325.0) for i, T in enumerate(state.Tg)])
    D_face = 0.5 * (D_cell[:-1, :] + D_cell[1:, :])

    x_safe = np.maximum(X_face, 1.0e-14)
    expected_Vd0 = -(D_face / x_safe) * dXdr
    expected_Vcd = -np.sum(Y_face * expected_Vd0, axis=1)
    expected_J = -rho_face[:, None] * Y_face * (expected_Vd0 + expected_Vcd[:, None])

    assert np.allclose(diff_pkg.Vd0_face_full, expected_Vd0)
    assert np.allclose(diff_pkg.Vcd_face, expected_Vcd)
    assert np.allclose(diff_pkg.J_diff_full, expected_J)


def test_internal_face_diffusive_flux_sums_to_zero() -> None:
    mesh = _make_stub_mesh()
    state = _make_state()
    gas_props = _StubGasProps()

    diff_pkg = build_gas_internal_diffusion_package(mesh, state, gas_props, p_env=101325.0)

    assert np.allclose(np.sum(diff_pkg.J_diff_full, axis=1), 0.0, atol=1.0e-10, rtol=0.0)


def test_farfield_face_diffusive_flux_sums_to_zero() -> None:
    mesh = _make_stub_mesh()
    state = _make_state()
    gas_props = _StubGasProps()
    Y_inf = np.array([0.18, 0.32, 0.50], dtype=np.float64)

    far_pkg = build_gas_farfield_boundary_flux_package(
        mesh,
        state,
        gas_props,
        p_env=101325.0,
        T_inf=1100.0,
        Yg_inf_full=Y_inf,
    )

    assert np.isclose(np.sum(far_pkg.J_diff_full), 0.0, atol=1.0e-10, rtol=0.0)


def test_internal_face_package_excludes_interface_and_farfield_faces() -> None:
    mesh = _make_stub_mesh()
    state = _make_state()
    gas_props = _StubGasProps()

    diff_pkg = build_gas_internal_diffusion_package(mesh, state, gas_props, p_env=101325.0)

    assert mesh.interface_face_index not in diff_pkg.face_indices
    assert (mesh.n_faces - 1) not in diff_pkg.face_indices
    assert np.array_equal(diff_pkg.face_indices, np.array([2, 3], dtype=np.int64))


def test_farfield_flux_package_is_built_separately() -> None:
    mesh = _make_stub_mesh()
    state = _make_state()
    gas_props = _StubGasProps()
    Y_inf = np.array([0.18, 0.32, 0.50], dtype=np.float64)

    diff_pkg = build_gas_internal_diffusion_package(mesh, state, gas_props, p_env=101325.0)
    far_pkg = build_gas_farfield_boundary_flux_package(
        mesh,
        state,
        gas_props,
        p_env=101325.0,
        T_inf=1100.0,
        Yg_inf_full=Y_inf,
    )

    assert far_pkg.face_index == mesh.n_faces - 1
    assert far_pkg.face_index not in diff_pkg.face_indices
    assert far_pkg.cell_index == state.n_gas_cells - 1


def test_flux_gas_uses_properties_layer_for_D_k_h() -> None:
    mesh = _make_stub_mesh()
    state = _make_state()
    gas_props = _StubGasProps()

    diff_pkg = build_gas_internal_diffusion_package(mesh, state, gas_props, p_env=101325.0)
    energy_pkg = build_gas_internal_energy_flux_package(mesh, state, gas_props, p_env=101325.0, diff_pkg=diff_pkg)
    _ = build_gas_farfield_boundary_flux_package(
        mesh,
        state,
        gas_props,
        p_env=101325.0,
        T_inf=1100.0,
        Yg_inf_full=np.array([0.18, 0.32, 0.50], dtype=np.float64),
    )

    assert gas_props.diffusivity_calls >= state.n_gas_cells + 1
    assert gas_props.conductivity_calls >= state.n_gas_cells + 1
    assert gas_props.species_enthalpy_calls >= state.n_gas_cells + 1
    assert energy_pkg.k_face.shape == (mesh.n_gas - 1,)


def test_trace_species_does_not_create_nan_in_Vd0() -> None:
    mesh = _make_stub_mesh()
    gas_props = _StubGasProps()
    state = _make_state(
        Yg_full=np.array(
            [
                [1.0e-16, 0.40, 0.60],
                [2.0e-16, 0.35, 0.65],
                [3.0e-16, 0.30, 0.70],
            ],
            dtype=np.float64,
        )
    )

    diff_pkg = build_gas_internal_diffusion_package(mesh, state, gas_props, p_env=101325.0)

    assert np.all(np.isfinite(diff_pkg.Vd0_face_full))
    assert np.all(np.isfinite(diff_pkg.J_diff_full))


def test_uniform_state_gives_zero_diffusive_and_thermal_flux() -> None:
    mesh = _make_stub_mesh()
    gas_props = _StubGasProps()
    state = _make_state(
        Tg=np.full(3, 950.0, dtype=np.float64),
        Yg_full=np.tile(np.array([[0.22, 0.28, 0.50]], dtype=np.float64), (3, 1)),
        rho_g=np.full(3, 1.15, dtype=np.float64),
    )
    Y_inf = np.array([0.22, 0.28, 0.50], dtype=np.float64)

    diff_pkg = build_gas_internal_diffusion_package(mesh, state, gas_props, p_env=101325.0)
    energy_pkg = build_gas_internal_energy_flux_package(mesh, state, gas_props, p_env=101325.0, diff_pkg=diff_pkg)
    far_pkg = build_gas_farfield_boundary_flux_package(
        mesh,
        state,
        gas_props,
        p_env=101325.0,
        T_inf=950.0,
        Yg_inf_full=Y_inf,
    )

    assert np.allclose(diff_pkg.J_diff_full, 0.0)
    assert np.allclose(energy_pkg.q_cond, 0.0)
    assert np.allclose(energy_pkg.q_species_diff, 0.0)
    assert np.allclose(energy_pkg.q_total, 0.0)
    assert np.allclose(far_pkg.J_diff_full, 0.0)
    assert np.isclose(far_pkg.q_cond, 0.0)
    assert np.isclose(far_pkg.q_species_diff, 0.0)
    assert np.isclose(far_pkg.q_total, 0.0)


def test_energy_flux_matches_cond_plus_species_enthalpy_diffusion_identity() -> None:
    mesh = _make_stub_mesh()
    state = _make_state()
    gas_props = _StubGasProps()

    diff_pkg = build_gas_internal_diffusion_package(mesh, state, gas_props, p_env=101325.0)
    energy_pkg = build_gas_internal_energy_flux_package(mesh, state, gas_props, p_env=101325.0, diff_pkg=diff_pkg)

    expected = energy_pkg.q_cond + np.sum(diff_pkg.J_diff_full * energy_pkg.h_face_full, axis=1)
    assert np.allclose(energy_pkg.q_species_diff, np.sum(diff_pkg.J_diff_full * energy_pkg.h_face_full, axis=1))
    assert np.allclose(energy_pkg.q_total, expected)
