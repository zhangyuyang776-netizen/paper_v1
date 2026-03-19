from __future__ import annotations

import numpy as np
import pytest

import cantera as ct

from properties.equilibrium import (
    InterfaceActivityModelError,
    InterfaceEquilibriumModelError,
    InterfaceEquilibriumValidationError,
    build_interface_equilibrium_model,
    compute_interface_equilibrium,
)
from properties.gas import build_gas_thermo_model
from properties.liquid import build_liquid_thermo_model
from properties.liquid_db import LiquidDBGlobalModels, LiquidDatabase, LiquidDBMeta, LiquidPureSpeciesRecord


def make_species_record(
    name: str,
    *,
    molecular_weight: float,
    cp_A: float,
    cp_T_range: tuple[float, float],
    rho_value: float,
    k_value: float,
    mu_value: float,
    boiling_temperature: float,
    hvap_ref: float,
    activity_model: str,
    unifac_groups: dict[str, float] | None,
) -> LiquidPureSpeciesRecord:
    def merino_coeffs(value: float) -> dict[str, float]:
        return {"A": 0.0, "B": 0.0, "C": 0.0, "D": float(np.log(value)), "E": 0.0, "F": 0.0}

    return LiquidPureSpeciesRecord(
        name=name,
        aliases=(),
        molecular_weight=molecular_weight,
        boiling_temperature=boiling_temperature,
        molar_volume=10.0,
        association_factor=1.0,
        T_ref=298.15,
        cp_model="shomate",
        cp_T_range=cp_T_range,
        cp_coeffs={"A": cp_A, "B": 0.0, "C": 0.0, "D": 0.0, "E": 0.0, "F": 0.0, "G": 0.0, "H": 0.0},
        hvap_ref=hvap_ref,
        Tc=600.0,
        hvap_model="watson",
        hvap_coeffs={},
        rho_model="merino_log_poly",
        rho_coeffs=merino_coeffs(rho_value),
        k_model="merino_log_poly",
        k_coeffs=merino_coeffs(k_value),
        mu_model="merino_log_poly",
        mu_coeffs=merino_coeffs(mu_value),
        activity_model=activity_model,
        unifac_groups=unifac_groups,
    )


def make_liquid_database(*, activity_model: str, with_unifac: bool) -> LiquidDatabase:
    water = make_species_record(
        "water",
        molecular_weight=0.01801528,
        cp_A=75.0,
        cp_T_range=(280.0, 650.0),
        rho_value=995.0,
        k_value=0.60,
        mu_value=0.9e-3,
        boiling_temperature=373.15,
        hvap_ref=2.2e6,
        activity_model=activity_model,
        unifac_groups=None if activity_model == "ideal" else {"OH": 2},
    )
    peroxide = make_species_record(
        "hydrogen_peroxide",
        molecular_weight=0.0340147,
        cp_A=110.0,
        cp_T_range=(280.0, 650.0),
        rho_value=1450.0,
        k_value=0.50,
        mu_value=1.5e-3,
        boiling_temperature=423.15,
        hvap_ref=1.0e6,
        activity_model=activity_model,
        unifac_groups=None if activity_model == "ideal" else {"OH": 2, "H2O2G": 1},
    )
    if with_unifac:
        unifac = {
            "enabled": True,
            "groups": {
                "OH": {"R": 1.0, "Q": 1.2},
                "H2O2G": {"R": 1.5, "Q": 1.1},
            },
            "interactions": {
                "OH": {"OH": 0.0, "H2O2G": 150.0},
                "H2O2G": {"OH": 50.0, "H2O2G": 0.0},
            },
        }
    else:
        unifac = None
    return LiquidDatabase(
        meta=LiquidDBMeta(
            file_type="liquid_properties_db",
            version=1,
            units={"molecular_weight": "kg/mol"},
            reference_T=298.15,
        ),
        global_models=LiquidDBGlobalModels(
            liquid_cp_default_model="shomate",
            liquid_density_default_model="merino_log_poly",
            liquid_conductivity_default_model="merino_log_poly",
            liquid_viscosity_default_model="merino_log_poly",
            liquid_diffusion_model="wilke_chang",
            liquid_mixture_density_model="merino_x_sqrt_rho",
            liquid_mixture_conductivity_model="filippov",
            liquid_mixture_viscosity_model="grunberg_nissan",
            activity_model_default=activity_model,
        ),
        unifac=unifac,
        species_by_name={"water": water, "hydrogen_peroxide": peroxide},
        alias_to_name={},
    )


def make_gas_backend():
    gas = ct.Solution("h2o2.yaml")
    species_names = tuple(gas.species_names)
    mw = np.asarray(gas.molecular_weights, dtype=np.float64) / 1000.0
    return build_gas_thermo_model(
        mechanism_path="h2o2.yaml",
        gas_species_full=species_names,
        molecular_weights=mw,
        closure_species="N2",
    )


def test_build_interface_equilibrium_model_rejects_invalid_mapping() -> None:
    liquid_db = make_liquid_database(activity_model="ideal", with_unifac=False)
    liquid = build_liquid_thermo_model(
        liquid_db=liquid_db,
        liquid_species_full=("water", "hydrogen_peroxide"),
        molecular_weights=np.array([0.01801528, 0.0340147], dtype=np.float64),
        p_env=101325.0,
    )
    gas = make_gas_backend()
    with pytest.raises(InterfaceEquilibriumValidationError, match="keys must exactly match"):
        build_interface_equilibrium_model(
            liquid_thermo=liquid,
            gas_thermo=gas,
            liquid_to_gas_species_map={"water": "H2O"},
            gas_closure_species="N2",
            reference_pressure=101325.0,
        )
    with pytest.raises(InterfaceEquilibriumValidationError, match="closure"):
        build_interface_equilibrium_model(
            liquid_thermo=liquid,
            gas_thermo=gas,
            liquid_to_gas_species_map={"water": "H2O", "hydrogen_peroxide": "N2"},
            gas_closure_species="N2",
            reference_pressure=101325.0,
        )


def test_compute_interface_equilibrium_single_component_ideal() -> None:
    liquid_db = make_liquid_database(activity_model="ideal", with_unifac=False)
    liquid = build_liquid_thermo_model(
        liquid_db=liquid_db,
        liquid_species_full=("water",),
        molecular_weights=np.array([0.01801528], dtype=np.float64),
        p_env=101325.0,
    )
    gas = make_gas_backend()
    model = build_interface_equilibrium_model(
        liquid_thermo=liquid,
        gas_thermo=gas,
        liquid_to_gas_species_map={"water": "H2O"},
        gas_closure_species="N2",
        reference_pressure=101325.0,
    )
    result = compute_interface_equilibrium(model, Ts=360.0, P=101325.0, Yl_if_full=np.array([1.0], dtype=np.float64))
    h2o_index = gas.species_names.index("H2O")
    n2_index = gas.species_names.index("N2")
    assert result.gamma_cond.shape == (1,)
    assert result.gamma_cond[0] == pytest.approx(1.0)
    assert result.Xg_eq_full.shape == (len(gas.species_names),)
    assert result.Yg_eq_full.shape == (len(gas.species_names),)
    assert np.isclose(np.sum(result.Xg_eq_full), 1.0)
    assert np.isclose(np.sum(result.Yg_eq_full), 1.0)
    assert result.Xg_eq_full[h2o_index] > 0.0
    assert result.Xg_eq_full[n2_index] >= 0.0
    assert all(result.Xg_eq_full[i] == 0.0 for i, name in enumerate(gas.species_names) if name not in {"H2O", "N2"})


def test_compute_interface_equilibrium_at_boiling_point_gives_zero_integral_and_psat_atm() -> None:
    liquid_db = make_liquid_database(activity_model="ideal", with_unifac=False)
    liquid = build_liquid_thermo_model(
        liquid_db=liquid_db,
        liquid_species_full=("water",),
        molecular_weights=np.array([0.01801528], dtype=np.float64),
        p_env=101325.0,
    )
    gas = make_gas_backend()
    model = build_interface_equilibrium_model(
        liquid_thermo=liquid,
        gas_thermo=gas,
        liquid_to_gas_species_map={"water": "H2O"},
        gas_closure_species="N2",
        reference_pressure=101325.0,
    )
    result = compute_interface_equilibrium(
        model,
        Ts=373.15,
        P=101325.0,
        Yl_if_full=np.array([1.0], dtype=np.float64),
    )
    assert result.psat_cond[0] == pytest.approx(101325.0, rel=1.0e-10, abs=1.0e-10)
    assert result.diagnostics["eq219_integral_cond"][0] == pytest.approx(0.0, abs=1.0e-10)


def test_compute_interface_equilibrium_multi_component_ideal_shapes_and_sums() -> None:
    liquid_db = make_liquid_database(activity_model="ideal", with_unifac=False)
    liquid = build_liquid_thermo_model(
        liquid_db=liquid_db,
        liquid_species_full=("water", "hydrogen_peroxide"),
        molecular_weights=np.array([0.01801528, 0.0340147], dtype=np.float64),
        p_env=101325.0,
    )
    gas = make_gas_backend()
    model = build_interface_equilibrium_model(
        liquid_thermo=liquid,
        gas_thermo=gas,
        liquid_to_gas_species_map={"water": "H2O", "hydrogen_peroxide": "H2O2"},
        gas_closure_species="N2",
        reference_pressure=101325.0,
    )
    result = compute_interface_equilibrium(
        model,
        Ts=380.0,
        P=101325.0,
        Yl_if_full=np.array([0.4, 0.6], dtype=np.float64),
    )
    assert result.gamma_cond.shape == (2,)
    assert result.psat_cond.shape == (2,)
    assert result.latent_cond.shape == (2,)
    assert result.Xg_eq_full.shape == (len(gas.species_names),)
    assert result.Yg_eq_full.shape == (len(gas.species_names),)
    assert np.all(np.isfinite(result.gamma_cond))
    assert np.all(result.gamma_cond > 0.0)
    assert np.isclose(np.sum(result.Xg_eq_full), 1.0)
    assert np.isclose(np.sum(result.Yg_eq_full), 1.0)
    assert result.diagnostics["n_condensables"] == 2


def test_compute_interface_equilibrium_unifac_branch_runs() -> None:
    liquid_db = make_liquid_database(activity_model="UNIFAC", with_unifac=True)
    liquid = build_liquid_thermo_model(
        liquid_db=liquid_db,
        liquid_species_full=("water", "hydrogen_peroxide"),
        molecular_weights=np.array([0.01801528, 0.0340147], dtype=np.float64),
        p_env=101325.0,
    )
    gas = make_gas_backend()
    model = build_interface_equilibrium_model(
        liquid_thermo=liquid,
        gas_thermo=gas,
        liquid_to_gas_species_map={"water": "H2O", "hydrogen_peroxide": "H2O2"},
        gas_closure_species="N2",
        reference_pressure=101325.0,
    )
    result = compute_interface_equilibrium(
        model,
        Ts=360.0,
        P=101325.0,
        Yl_if_full=np.array([0.5, 0.5], dtype=np.float64),
    )
    assert np.all(np.isfinite(result.gamma_cond))
    assert np.all(result.gamma_cond > 0.0)
    assert result.diagnostics["activity_model"] == "UNIFAC"


def test_compute_interface_equilibrium_rejects_condensable_sum_above_unity() -> None:
    liquid_db = make_liquid_database(activity_model="UNIFAC", with_unifac=True)
    liquid = build_liquid_thermo_model(
        liquid_db=liquid_db,
        liquid_species_full=("water", "hydrogen_peroxide"),
        molecular_weights=np.array([0.01801528, 0.0340147], dtype=np.float64),
        p_env=101325.0,
    )
    gas = make_gas_backend()
    model = build_interface_equilibrium_model(
        liquid_thermo=liquid,
        gas_thermo=gas,
        liquid_to_gas_species_map={"water": "H2O", "hydrogen_peroxide": "H2O2"},
        gas_closure_species="N2",
        reference_pressure=101325.0,
    )
    with pytest.raises(InterfaceEquilibriumModelError, match="exceed unity"):
        compute_interface_equilibrium(
            model,
            Ts=390.0,
            P=101325.0,
            Yl_if_full=np.array([0.5, 0.5], dtype=np.float64),
        )


def test_compute_interface_equilibrium_unifac_requires_data() -> None:
    liquid_db = make_liquid_database(activity_model="UNIFAC", with_unifac=False)
    liquid = build_liquid_thermo_model(
        liquid_db=liquid_db,
        liquid_species_full=("water", "hydrogen_peroxide"),
        molecular_weights=np.array([0.01801528, 0.0340147], dtype=np.float64),
        p_env=101325.0,
    )
    gas = make_gas_backend()
    model = build_interface_equilibrium_model(
        liquid_thermo=liquid,
        gas_thermo=gas,
        liquid_to_gas_species_map={"water": "H2O", "hydrogen_peroxide": "H2O2"},
        gas_closure_species="N2",
        reference_pressure=101325.0,
    )
    with pytest.raises(InterfaceActivityModelError, match="no unifac data"):
        compute_interface_equilibrium(
            model,
            Ts=390.0,
            P=101325.0,
            Yl_if_full=np.array([0.5, 0.5], dtype=np.float64),
        )


def test_compute_interface_equilibrium_rejects_invalid_liquid_input() -> None:
    liquid_db = make_liquid_database(activity_model="ideal", with_unifac=False)
    liquid = build_liquid_thermo_model(
        liquid_db=liquid_db,
        liquid_species_full=("water",),
        molecular_weights=np.array([0.01801528], dtype=np.float64),
        p_env=101325.0,
    )
    gas = make_gas_backend()
    model = build_interface_equilibrium_model(
        liquid_thermo=liquid,
        gas_thermo=gas,
        liquid_to_gas_species_map={"water": "H2O"},
        gas_closure_species="N2",
        reference_pressure=101325.0,
    )
    with pytest.raises(InterfaceEquilibriumValidationError, match="sum to 1"):
        compute_interface_equilibrium(model, Ts=360.0, P=101325.0, Yl_if_full=np.array([0.9], dtype=np.float64))
    with pytest.raises(InterfaceEquilibriumValidationError, match="temperature range"):
        compute_interface_equilibrium(model, Ts=700.0, P=101325.0, Yl_if_full=np.array([1.0], dtype=np.float64))
