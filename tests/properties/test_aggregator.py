from __future__ import annotations

import numpy as np
import pytest

import cantera as ct

from core.types import InterfaceState, Mesh1D, RegionSlices, State
from properties.aggregator import AggregatorValidationError, build_bulk_props
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
) -> LiquidPureSpeciesRecord:
    def merino_coeffs(value: float) -> dict[str, float]:
        return {"A": 0.0, "B": 0.0, "C": 0.0, "D": float(np.log(value)), "E": 0.0, "F": 0.0}

    return LiquidPureSpeciesRecord(
        name=name,
        aliases=(),
        molecular_weight=molecular_weight,
        boiling_temperature=350.0,
        molar_volume=10.0,
        association_factor=1.0,
        T_ref=298.15,
        cp_model="shomate",
        cp_T_range=cp_T_range,
        cp_coeffs={"A": cp_A, "B": 0.0, "C": 0.0, "D": 0.0, "E": 0.0, "F": 0.0, "G": 0.0, "H": 0.0},
        hvap_ref=1.0e5,
        Tc=500.0,
        hvap_model="watson",
        hvap_coeffs={},
        rho_model="merino_log_poly",
        rho_coeffs=merino_coeffs(rho_value),
        k_model="merino_log_poly",
        k_coeffs=merino_coeffs(k_value),
        mu_model="merino_log_poly",
        mu_coeffs=merino_coeffs(mu_value),
        activity_model=None,
        unifac_groups=None,
    )


def make_liquid_database(*, multi: bool) -> LiquidDatabase:
    ethanol = make_species_record(
        "ethanol",
        molecular_weight=0.04607,
        cp_A=100.0,
        cp_T_range=(250.0, 600.0),
        rho_value=780.0,
        k_value=0.16,
        mu_value=1.2e-3,
    )
    species_by_name: dict[str, LiquidPureSpeciesRecord] = {"ethanol": ethanol}
    if multi:
        water = make_species_record(
            "water",
            molecular_weight=0.01801528,
            cp_A=75.0,
            cp_T_range=(280.0, 550.0),
            rho_value=995.0,
            k_value=0.60,
            mu_value=0.9e-3,
        )
        species_by_name["water"] = water

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
            activity_model_default=None,
        ),
        unifac=None,
        species_by_name=species_by_name,
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


def make_state(*, liquid_multi: bool, gas_species_names: tuple[str, ...]) -> State:
    if liquid_multi:
        Tl = np.array([300.0, 320.0], dtype=np.float64)
        Yl_full = np.array([[0.3, 0.7], [0.6, 0.4]], dtype=np.float64)
        Ys_l_full = np.array([0.5, 0.5], dtype=np.float64)
    else:
        Tl = np.array([305.0, 325.0], dtype=np.float64)
        Yl_full = np.ones((2, 1), dtype=np.float64)
        Ys_l_full = np.array([1.0], dtype=np.float64)

    def gas_y(h2: float, o2: float, n2: float) -> np.ndarray:
        Y = np.zeros(len(gas_species_names), dtype=np.float64)
        Y[gas_species_names.index("H2")] = h2
        Y[gas_species_names.index("O2")] = o2
        Y[gas_species_names.index("N2")] = n2
        return Y

    Tg = np.array([900.0, 1100.0], dtype=np.float64)
    Yg_full = np.vstack([gas_y(0.2, 0.3, 0.5), gas_y(0.25, 0.25, 0.5)])
    Ys_g_full = gas_y(0.1, 0.2, 0.7)

    return State(
        Tl=Tl,
        Yl_full=Yl_full,
        Tg=Tg,
        Yg_full=Yg_full,
        interface=InterfaceState(Ts=330.0, mpp=0.0, Ys_g_full=Ys_g_full, Ys_l_full=Ys_l_full),
    )


def test_build_bulk_props_single_component_liquid() -> None:
    liquid_db = make_liquid_database(multi=False)
    liquid = build_liquid_thermo_model(
        liquid_db=liquid_db,
        liquid_species_full=("ethanol",),
        molecular_weights=np.array([0.04607], dtype=np.float64),
        p_env=101325.0,
    )
    gas = make_gas_backend()
    grid = make_dummy_mesh(n_liq=2, n_gas=2)
    state = make_state(liquid_multi=False, gas_species_names=gas.species_names)

    props = build_bulk_props(state=state, grid=grid, liquid_thermo=liquid, gas_thermo=gas)

    assert props.rho_l.shape == (2,)
    assert props.cp_l.shape == (2,)
    assert props.h_l.shape == (2,)
    assert props.k_l.shape == (2,)
    assert props.mu_l.shape == (2,)
    assert props.D_l is None
    assert props.rho_g.shape == (2,)
    assert props.D_g.shape == (2, gas.n_species)
    assert props.diagnostics["all_finite"] is True
    assert not hasattr(props, "Ts")
    assert not hasattr(props, "mpp")


def test_build_bulk_props_multicomponent_liquid_and_multispecies_gas() -> None:
    liquid_db = make_liquid_database(multi=True)
    liquid = build_liquid_thermo_model(
        liquid_db=liquid_db,
        liquid_species_full=("ethanol", "water"),
        molecular_weights=np.array([0.04607, 0.01801528], dtype=np.float64),
        p_env=101325.0,
    )
    gas = make_gas_backend()
    grid = make_dummy_mesh(n_liq=2, n_gas=2)
    state = make_state(liquid_multi=True, gas_species_names=gas.species_names)

    props = build_bulk_props(state=state, grid=grid, liquid_thermo=liquid, gas_thermo=gas)

    assert props.D_l.shape == (2, 2)
    assert props.D_g.shape == (2, gas.n_species)
    assert props.diagnostics["n_liq_cells"] == 2
    assert props.diagnostics["n_gas_cells"] == 2
    assert props.diagnostics["liquid_species_count"] == 2
    assert props.diagnostics["gas_species_count"] == gas.n_species
    assert props.diagnostics["min_rho_l"] > 0.0
    assert props.diagnostics["min_cp_g"] > 0.0


def test_build_bulk_props_accepts_explicit_gas_pressure_vector() -> None:
    liquid_db = make_liquid_database(multi=False)
    liquid = build_liquid_thermo_model(
        liquid_db=liquid_db,
        liquid_species_full=("ethanol",),
        molecular_weights=np.array([0.04607], dtype=np.float64),
        p_env=101325.0,
    )
    gas = make_gas_backend()
    grid = make_dummy_mesh(n_liq=2, n_gas=2)
    state = make_state(liquid_multi=False, gas_species_names=gas.species_names)

    props_default = build_bulk_props(state=state, grid=grid, liquid_thermo=liquid, gas_thermo=gas)
    props_high_p = build_bulk_props(
        state=state,
        grid=grid,
        liquid_thermo=liquid,
        gas_thermo=gas,
        gas_pressure=np.array([2.0 * gas.reference_pressure, 2.0 * gas.reference_pressure], dtype=np.float64),
    )

    assert np.all(props_high_p.rho_g > props_default.rho_g)
    assert props_high_p.diagnostics["min_gas_pressure"] == pytest.approx(2.0 * gas.reference_pressure)
    assert props_high_p.diagnostics["max_gas_pressure"] == pytest.approx(2.0 * gas.reference_pressure)


def test_build_bulk_props_rejects_state_grid_mismatch() -> None:
    liquid_db = make_liquid_database(multi=False)
    liquid = build_liquid_thermo_model(
        liquid_db=liquid_db,
        liquid_species_full=("ethanol",),
        molecular_weights=np.array([0.04607], dtype=np.float64),
        p_env=101325.0,
    )
    gas = make_gas_backend()
    grid = make_dummy_mesh(n_liq=1, n_gas=2)
    state = make_state(liquid_multi=False, gas_species_names=gas.species_names)

    with pytest.raises(AggregatorValidationError, match="grid.n_liq"):
        build_bulk_props(state=state, grid=grid, liquid_thermo=liquid, gas_thermo=gas)
