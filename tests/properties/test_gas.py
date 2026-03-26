from __future__ import annotations

import numpy as np
import pytest

import cantera as ct

from core.state_recovery import recover_state_from_contents
from core.types import ConservativeContents, InterfaceState, Mesh1D, RecoveryConfig, RegionSlices, SpeciesMaps
from properties.gas import (
    GasThermoModelError,
    GasThermoValidationError,
    build_gas_thermo_model,
)


class FakeLiquidThermo:
    def __init__(self, cp: float) -> None:
        self.cp = float(cp)

    def enthalpy_mass(self, T: float, Y_full: np.ndarray) -> float:
        return self.cp * float(T)


def make_mechanism_contract() -> tuple[str, tuple[str, ...], np.ndarray]:
    gas = ct.Solution("h2o2.yaml")
    return "h2o2.yaml", tuple(gas.species_names), np.asarray(gas.molecular_weights, dtype=np.float64) / 1000.0


def make_recovery_config() -> RecoveryConfig:
    return RecoveryConfig(
        rho_min=1.0e-12,
        m_min=1.0e-20,
        species_recovery_eps_abs=1.0e-14,
        Y_sum_tol=1.0e-10,
        Y_hard_tol=1.0e-6,
        h_abs_tol=1.0e-8,
        h_rel_tol=1.0e-10,
        h_check_tol=1.0e-8,
        T_step_tol=1.0e-8,
        T_min_l=280.0,
        T_max_l=540.0,
        T_min_g=250.0,
        T_max_g=3000.0,
        liquid_h_inv_max_iter=200,
        cp_min=1.0,
        gas_h_inv_max_iter=200,
        use_cantera_hpy_first=True,
    )


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


def make_species_maps(gas_species_full: tuple[str, ...]) -> SpeciesMaps:
    gas_closure = "N2"
    gas_active = tuple(name for name in gas_species_full if name != gas_closure)
    gas_full_to_reduced = np.full(len(gas_species_full), -1, dtype=np.int64)
    reduced_to_full: list[int] = []
    next_index = 0
    for i, name in enumerate(gas_species_full):
        if name == gas_closure:
            continue
        gas_full_to_reduced[i] = next_index
        reduced_to_full.append(i)
        next_index += 1
    return SpeciesMaps(
        liq_full_names=("ethanol",),
        liq_active_names=(),
        liq_closure_name=None,
        gas_full_names=gas_species_full,
        gas_active_names=gas_active,
        gas_closure_name=gas_closure,
        liq_full_to_reduced=np.array([-1], dtype=np.int64),
        liq_reduced_to_full=np.array([], dtype=np.int64),
        gas_full_to_reduced=gas_full_to_reduced,
        gas_reduced_to_full=np.array(reduced_to_full, dtype=np.int64),
        liq_full_to_gas_full=np.array([gas_species_full.index("H2O")], dtype=np.int64),
    )


def make_interface_seed(species_maps: SpeciesMaps) -> InterfaceState:
    return InterfaceState(
        Ts=320.0,
        mpp=0.0,
        Ys_g_full=np.full(species_maps.n_gas_full, 1.0 / species_maps.n_gas_full, dtype=np.float64),
        Ys_l_full=np.array([1.0], dtype=np.float64),
    )


def make_gas_composition(species_names: tuple[str, ...]) -> np.ndarray:
    Y = np.zeros(len(species_names), dtype=np.float64)
    Y[species_names.index("H2")] = 0.2
    Y[species_names.index("O2")] = 0.3
    Y[species_names.index("N2")] = 0.5
    return Y


def test_build_gas_thermo_model_success() -> None:
    mechanism_path, species_names, mw = make_mechanism_contract()
    model = build_gas_thermo_model(
        mechanism_path=mechanism_path,
        gas_species_full=species_names,
        molecular_weights=mw,
        closure_species="N2",
    )
    assert model.species_names == species_names
    assert model.closure_species == "N2"
    assert model.transport_model is not None


def test_build_gas_thermo_model_rejects_species_order_mismatch() -> None:
    mechanism_path, species_names, mw = make_mechanism_contract()
    bad_names = (species_names[1], species_names[0], *species_names[2:])
    with pytest.raises(GasThermoModelError, match="exactly match"):
        build_gas_thermo_model(
            mechanism_path=mechanism_path,
            gas_species_full=bad_names,
            molecular_weights=mw,
            closure_species="N2",
        )


def test_build_gas_thermo_model_rejects_molecular_weights_mismatch() -> None:
    mechanism_path, species_names, mw = make_mechanism_contract()
    bad_mw = mw.copy()
    bad_mw[0] *= 1.1
    with pytest.raises(GasThermoModelError, match="molecular_weights"):
        build_gas_thermo_model(
            mechanism_path=mechanism_path,
            gas_species_full=species_names,
            molecular_weights=bad_mw,
            closure_species="N2",
        )


def test_build_gas_thermo_model_rejects_missing_closure_species() -> None:
    mechanism_path, species_names, mw = make_mechanism_contract()
    with pytest.raises(GasThermoValidationError, match="closure_species"):
        build_gas_thermo_model(
            mechanism_path=mechanism_path,
            gas_species_full=species_names,
            molecular_weights=mw,
            closure_species="CO2",
        )


def test_gas_scalar_properties_are_finite_and_positive() -> None:
    mechanism_path, species_names, mw = make_mechanism_contract()
    model = build_gas_thermo_model(
        mechanism_path=mechanism_path,
        gas_species_full=species_names,
        molecular_weights=mw,
        closure_species="N2",
    )
    Y = make_gas_composition(species_names)
    T = 900.0
    P = 101325.0
    assert model.density_mass(T, Y, P) > 0.0
    assert model.cp_mass(T, Y, P) > 0.0
    assert np.isfinite(model.enthalpy_mass(T, Y, P))
    assert model.conductivity(T, Y, P) > 0.0
    assert model.viscosity(T, Y, P) > 0.0
    D = model.diffusivity(T, Y, P)
    assert D.shape == (len(species_names),)
    assert np.all(D > 0.0)
    h_species = model.species_enthalpies_mass(T)
    assert h_species.shape == (len(species_names),)
    assert np.all(np.isfinite(h_species))


def test_gas_full_composition_validation_is_strict() -> None:
    mechanism_path, species_names, mw = make_mechanism_contract()
    model = build_gas_thermo_model(
        mechanism_path=mechanism_path,
        gas_species_full=species_names,
        molecular_weights=mw,
        closure_species="N2",
    )
    Y = make_gas_composition(species_names)
    with pytest.raises(GasThermoValidationError, match="non-negative"):
        model.enthalpy_mass(900.0, np.where(np.arange(len(Y)) == 0, -0.1, Y), 101325.0)
    with pytest.raises(GasThermoValidationError, match="sum to 1"):
        model.cp_mass(900.0, Y * 0.9, 101325.0)
    with pytest.raises(GasThermoValidationError, match="length"):
        model.density_mass(900.0, Y[:-1], 101325.0)


def test_gas_batch_methods_match_scalar_results() -> None:
    mechanism_path, species_names, mw = make_mechanism_contract()
    model = build_gas_thermo_model(
        mechanism_path=mechanism_path,
        gas_species_full=species_names,
        molecular_weights=mw,
        closure_species="N2",
    )
    Y1 = make_gas_composition(species_names)
    Y2 = Y1.copy()
    Y2[species_names.index("H2")] = 0.25
    Y2[species_names.index("O2")] = 0.25
    Y2[species_names.index("N2")] = 0.50
    T = np.array([900.0, 1100.0], dtype=np.float64)
    Y = np.vstack([Y1, Y2])
    P = np.array([101325.0, 202650.0], dtype=np.float64)
    rho_batch = model.density_mass_batch(T, Y, P)
    cp_batch = model.cp_mass_batch(T, Y, P)
    h_batch = model.enthalpy_mass_batch(T, Y, P)
    rho_scalar = np.array([model.density_mass(T[i], Y[i, :], P[i]) for i in range(2)], dtype=np.float64)
    cp_scalar = np.array([model.cp_mass(T[i], Y[i, :], P[i]) for i in range(2)], dtype=np.float64)
    h_scalar = np.array([model.enthalpy_mass(T[i], Y[i, :], P[i]) for i in range(2)], dtype=np.float64)
    assert np.allclose(rho_batch, rho_scalar)
    assert np.allclose(cp_batch, cp_scalar)
    assert np.allclose(h_batch, h_scalar)


def test_gas_batch_methods_accept_scalar_pressure() -> None:
    mechanism_path, species_names, mw = make_mechanism_contract()
    model = build_gas_thermo_model(
        mechanism_path=mechanism_path,
        gas_species_full=species_names,
        molecular_weights=mw,
        closure_species="N2",
    )
    Y = np.vstack([make_gas_composition(species_names), make_gas_composition(species_names)])
    T = np.array([850.0, 950.0], dtype=np.float64)
    rho = model.density_mass_batch(T, Y, 101325.0)
    assert rho.shape == (2,)
    assert np.all(rho > 0.0)


def test_gas_enthalpy_roundtrip_with_state_recovery() -> None:
    mechanism_path, species_names, mw = make_mechanism_contract()
    gas_thermo = build_gas_thermo_model(
        mechanism_path=mechanism_path,
        gas_species_full=species_names,
        molecular_weights=mw,
        closure_species="N2",
    )
    species_maps = make_species_maps(species_names)
    mesh = make_dummy_mesh(n_liq=1, n_gas=2)
    Yg = np.vstack([make_gas_composition(species_names), make_gas_composition(species_names)])
    Tg_true = np.array([900.0, 1100.0], dtype=np.float64)
    P = 101325.0
    rho_g = gas_thermo.density_mass_batch(Tg_true, Yg, P)
    mass_g = rho_g * mesh.volumes[mesh.region_slices.gas_all]
    species_mass_g = mass_g[:, None] * Yg
    enthalpy_g = mass_g * gas_thermo.enthalpy_mass_batch(Tg_true, Yg, P)

    mass_l = np.array([780.0], dtype=np.float64)
    species_mass_l = np.array([[780.0]], dtype=np.float64)
    enthalpy_l = mass_l * np.array([300.0 * 2.0], dtype=np.float64)

    contents = ConservativeContents(
        mass_l=mass_l,
        species_mass_l=species_mass_l,
        enthalpy_l=enthalpy_l,
        mass_g=mass_g,
        species_mass_g=species_mass_g,
        enthalpy_g=enthalpy_g,
    )
    state = recover_state_from_contents(
        contents=contents,
        mesh=mesh,
        species_maps=species_maps,
        recovery_cfg=make_recovery_config(),
        liquid_thermo=FakeLiquidThermo(cp=2.0),
        gas_thermo=gas_thermo,
        interface_seed=make_interface_seed(species_maps),
    )
    assert np.allclose(state.Tg, Tg_true, atol=1.0e-6)
