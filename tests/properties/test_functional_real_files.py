from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import cantera as ct
import numpy as np

from core.config_loader import load_and_validate_config
from core.grid import build_grid_and_metrics
from core.layout import build_layout
from core.preprocess import normalize_config
from core.state_pack import pack_state_to_array, unpack_array_to_state
from core.types import GeometryState, InterfaceState, State
from properties.aggregator import build_bulk_props
from properties.equilibrium import (
    build_interface_equilibrium_model,
    compute_interface_equilibrium,
)
from properties.gas import build_gas_thermo_model
from properties.liquid import build_liquid_thermo_model
from properties.liquid_db import build_liquid_database


TESTS_DIR = Path(__file__).resolve().parents[1]
PAPER_ROOT = TESTS_DIR.parent
CASE_CONFIG_PATH = PAPER_ROOT / "cases" / "config.yaml"
OUT_DIR = TESTS_DIR / "out"


def _to_serializable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _to_serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_serializable(item) for item in value]
    return value


def _write_report(filename: str, payload: dict[str, Any]) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    target = OUT_DIR / filename
    target.write_text(json.dumps(_to_serializable(payload), indent=2), encoding="utf-8")
    return target


def _base_raw_config() -> dict[str, Any]:
    return load_and_validate_config(CASE_CONFIG_PATH)


def _run_config_from_raw(raw_cfg: dict[str, Any]):
    return normalize_config(raw_cfg, source_path=CASE_CONFIG_PATH)


def _liquid_molecular_weights(run_cfg, liquid_db) -> np.ndarray:
    return np.array(
        [liquid_db.species_by_name[name].molecular_weight for name in run_cfg.species_maps.liq_full_names],
        dtype=np.float64,
    )


def _gas_molecular_weights(run_cfg) -> np.ndarray:
    gas = ct.Solution(str(run_cfg.paths.gas_mechanism_path))
    return np.asarray(gas.molecular_weights, dtype=np.float64) / 1000.0


def _build_models(run_cfg):
    liquid_db = build_liquid_database(run_cfg)
    liquid_thermo = build_liquid_thermo_model(
        liquid_db=liquid_db,
        liquid_species_full=run_cfg.species_maps.liq_full_names,
        molecular_weights=_liquid_molecular_weights(run_cfg, liquid_db),
        p_env=run_cfg.pressure,
    )
    gas_thermo = build_gas_thermo_model(
        mechanism_path=run_cfg.paths.gas_mechanism_path,
        gas_species_full=run_cfg.species_maps.gas_full_names,
        molecular_weights=_gas_molecular_weights(run_cfg),
        closure_species=run_cfg.species.gas_closure_species,
    )
    return liquid_db, liquid_thermo, gas_thermo


def _build_geometry_and_layout(run_cfg):
    geometry = GeometryState(
        t=run_cfg.time_stepper.t0,
        dt=run_cfg.time_stepper.dt_start,
        a=run_cfg.mesh.a0,
        dot_a=-1.0e-2,
        r_end=run_cfg.mesh.r_end,
        step_index=0,
        outer_iter_index=0,
    )
    mesh, metrics = build_grid_and_metrics(run_cfg, geometry)
    layout = build_layout(run_cfg, mesh)
    return geometry, mesh, metrics, layout


def _make_single_component_state(run_cfg, mesh, interface: InterfaceState) -> State:
    Tl = np.linspace(300.0, 340.0, mesh.n_liq, dtype=np.float64)
    Yl_full = np.ones((mesh.n_liq, 1), dtype=np.float64)

    Tg = np.linspace(
        run_cfg.initialization.gas_temperature,
        run_cfg.initialization.gas_temperature + 50.0,
        mesh.n_gas,
        dtype=np.float64,
    )
    Yg_full = np.zeros((mesh.n_gas, run_cfg.species_maps.n_gas_full), dtype=np.float64)
    gas_names = run_cfg.species_maps.gas_full_names
    Yg_full[:, gas_names.index("C2H5OH")] = 0.02
    Yg_full[:, gas_names.index("O2")] = 0.21
    Yg_full[:, gas_names.index("N2")] = 0.77

    return State(
        Tl=Tl,
        Yl_full=Yl_full,
        Tg=Tg,
        Yg_full=Yg_full,
        interface=interface,
        time=run_cfg.time_stepper.t0,
        state_id="single_ethanol_baseline",
    )


def _make_multicomponent_state(run_cfg, mesh, interface: InterfaceState) -> State:
    Tl = np.linspace(300.0, 340.0, mesh.n_liq, dtype=np.float64)
    x = np.linspace(0.0, 1.0, mesh.n_liq, dtype=np.float64)
    ethanol = 0.7 - 0.2 * x
    water = 1.0 - ethanol
    Yl_full = np.column_stack([ethanol, water])

    Tg = np.linspace(950.0, 1050.0, mesh.n_gas, dtype=np.float64)
    Yg_full = np.zeros((mesh.n_gas, run_cfg.species_maps.n_gas_full), dtype=np.float64)
    gas_names = run_cfg.species_maps.gas_full_names
    Yg_full[:, gas_names.index("C2H5OH")] = 0.03
    Yg_full[:, gas_names.index("H2O")] = 0.04
    Yg_full[:, gas_names.index("O2")] = 0.21
    Yg_full[:, gas_names.index("N2")] = 0.72

    return State(
        Tl=Tl,
        Yl_full=Yl_full,
        Tg=Tg,
        Yg_full=Yg_full,
        interface=interface,
        time=run_cfg.time_stepper.t0,
        state_id="ethanol_water_500kpa",
    )


def _roundtrip_report(state: State, layout, species_maps) -> dict[str, float]:
    vec = pack_state_to_array(state, layout, species_maps)
    state_rt = unpack_array_to_state(vec, layout, species_maps, time=state.time, state_id=state.state_id)
    return {
        "vector_size": int(vec.shape[0]),
        "max_abs_Tl_error": float(np.max(np.abs(state_rt.Tl - state.Tl))),
        "max_abs_Yl_error": float(np.max(np.abs(state_rt.Yl_full - state.Yl_full))),
        "max_abs_Tg_error": float(np.max(np.abs(state_rt.Tg - state.Tg))),
        "max_abs_Yg_error": float(np.max(np.abs(state_rt.Yg_full - state.Yg_full))),
    }


def _build_pressure_bank_scan_report() -> dict[str, Any]:
    raw_cfg = _base_raw_config()
    pressures = [101325.0, 5.0e5, 1.0e6, 2.0e6]
    report_rows: list[dict[str, Any]] = []
    for pressure in pressures:
        scenario_cfg = deepcopy(raw_cfg)
        scenario_cfg["initialization"]["gas_pressure"] = pressure
        run_cfg = _run_config_from_raw(scenario_cfg)
        liquid_db, liquid_thermo, gas_thermo = _build_models(run_cfg)
        Y_ethanol = run_cfg.initialization.liquid_y_full_0
        gas_y = run_cfg.initialization.gas_y_full_0
        T_liq = min(300.0, liquid_thermo.valid_temperature_range()[1] - 1.0e-6)
        T_gas = run_cfg.initialization.gas_temperature
        report_rows.append(
            {
                "pressure": pressure,
                "selected_bank_pressures": liquid_thermo.selected_bank_pressures.tolist(),
                "liquid_valid_temperature_range": list(liquid_thermo.valid_temperature_range()),
                "liquid_cp_mass": float(liquid_thermo.cp_mass(T_liq, Y_ethanol)),
                "liquid_density_mass": float(liquid_thermo.density_mass(T_liq, Y_ethanol)),
                "liquid_conductivity": float(liquid_thermo.conductivity(T_liq, Y_ethanol)),
                "liquid_viscosity": float(liquid_thermo.viscosity(T_liq, Y_ethanol)),
                "gas_cp_mass": float(gas_thermo.cp_mass(T_gas, gas_y, pressure)),
                "gas_density_mass": float(gas_thermo.density_mass(T_gas, gas_y, pressure)),
                "gas_viscosity": float(gas_thermo.viscosity(T_gas, gas_y, pressure)),
            }
        )
    return {"scenario": "pressure_bank_scan", "rows": report_rows}


def test_functional_single_component_case_and_bank_scan() -> None:
    run_cfg = _run_config_from_raw(_base_raw_config())
    liquid_db, liquid_thermo, gas_thermo = _build_models(run_cfg)
    geometry, mesh, metrics, layout = _build_geometry_and_layout(run_cfg)

    interface = InterfaceState(
        Ts=320.0,
        mpp=0.0,
        Ys_g_full=run_cfg.initialization.y_vap_if0_gas_full,
        Ys_l_full=run_cfg.initialization.liquid_y_full_0,
    )
    state = _make_single_component_state(run_cfg, mesh, interface)
    props = build_bulk_props(
        state=state,
        grid=mesh,
        liquid_thermo=liquid_thermo,
        gas_thermo=gas_thermo,
        gas_pressure=run_cfg.pressure,
    )

    roundtrip = _roundtrip_report(state, layout, run_cfg.species_maps)
    report = {
        "scenario": "single_ethanol_baseline",
        "case_name": run_cfg.case_name,
        "paths": {
            "liquid_database_path": run_cfg.paths.liquid_database_path,
            "gas_mechanism_path": run_cfg.paths.gas_mechanism_path,
        },
        "species": {
            "liquid_species_full": run_cfg.species_maps.liq_full_names,
            "gas_species_full": run_cfg.species_maps.gas_full_names,
        },
        "geometry": {
            "a": geometry.a,
            "r_end": geometry.r_end,
            "n_liq_cells": mesh.n_liq,
            "n_gas_cells": mesh.n_gas,
            "interface_face_index": metrics.interface_face_index,
        },
        "layout": {
            "unknowns_profile": layout.unknowns_profile,
            "total_size": layout.total_size,
            "liq_cell_width": layout.liq_cell_width,
            "gas_cell_width": layout.gas_cell_width,
        },
        "liquid_backend": {
            "selected_bank_pressures": liquid_thermo.selected_bank_pressures.tolist(),
            "valid_temperature_range": list(liquid_thermo.valid_temperature_range()),
            "sample_cp_mass": float(liquid_thermo.cp_mass(298.15, run_cfg.initialization.liquid_y_full_0)),
            "sample_density_mass": float(
                liquid_thermo.density_mass(298.15, run_cfg.initialization.liquid_y_full_0)
            ),
        },
        "gas_backend": {
            "reference_pressure": gas_thermo.reference_pressure,
            "sample_cp_mass": float(
                gas_thermo.cp_mass(
                    run_cfg.initialization.gas_temperature,
                    run_cfg.initialization.gas_y_full_0,
                    run_cfg.pressure,
                )
            ),
            "sample_density_mass": float(
                gas_thermo.density_mass(
                    run_cfg.initialization.gas_temperature,
                    run_cfg.initialization.gas_y_full_0,
                    run_cfg.pressure,
                )
            ),
        },
        "state_pack_roundtrip": roundtrip,
        "bulk_props": {
            "diagnostics": props.diagnostics,
            "D_l_is_none": props.D_l is None,
            "D_g_shape": list(props.D_g.shape) if props.D_g is not None else None,
            "min_mu_l": float(np.min(props.mu_l)),
            "min_mu_g": float(np.min(props.mu_g)),
        },
    }

    report_path = _write_report("functional_single_ethanol_baseline.json", report)
    bank_scan_path = _write_report("functional_pressure_bank_scan.json", _build_pressure_bank_scan_report())

    assert report_path.exists()
    assert bank_scan_path.exists()
    assert liquid_thermo.selected_bank_pressures.tolist() == [101320.0]
    assert roundtrip["max_abs_Tl_error"] == 0.0
    assert roundtrip["max_abs_Yg_error"] == 0.0
    assert props.diagnostics["all_finite"] is True
    assert props.D_l is None


def test_functional_multicomponent_equilibrium_and_bulk_props() -> None:
    raw_cfg = _base_raw_config()
    raw_cfg["initialization"]["gas_pressure"] = 5.0e5
    raw_cfg["species"]["liquid_closure_species"] = "water"
    raw_cfg["species"]["liquid_to_gas_species_map"] = {"ethanol": "C2H5OH", "water": "H2O"}
    raw_cfg["initialization"]["liquid_composition"] = {"ethanol": 0.7, "water": 0.3}
    raw_cfg["initialization"]["Y_vap_if0"] = {"C2H5OH": 0.05, "H2O": 0.05}

    run_cfg = _run_config_from_raw(raw_cfg)
    liquid_db, liquid_thermo, gas_thermo = _build_models(run_cfg)
    geometry, mesh, metrics, layout = _build_geometry_and_layout(run_cfg)

    equilibrium_model = build_interface_equilibrium_model(
        liquid_thermo=liquid_thermo,
        gas_thermo=gas_thermo,
        liquid_to_gas_species_map=run_cfg.species.liquid_to_gas_species_map,
        gas_closure_species=run_cfg.species.gas_closure_species,
        reference_pressure=run_cfg.pressure,
    )
    eq_result = compute_interface_equilibrium(
        equilibrium_model,
        Ts=340.0,
        P=run_cfg.pressure,
        Yl_if_full=run_cfg.initialization.liquid_y_full_0,
    )

    interface = InterfaceState(
        Ts=340.0,
        mpp=0.0,
        Ys_g_full=eq_result.Yg_eq_full,
        Ys_l_full=run_cfg.initialization.liquid_y_full_0,
    )
    state = _make_multicomponent_state(run_cfg, mesh, interface)
    props = build_bulk_props(
        state=state,
        grid=mesh,
        liquid_thermo=liquid_thermo,
        gas_thermo=gas_thermo,
        gas_pressure=run_cfg.pressure,
    )
    roundtrip = _roundtrip_report(state, layout, run_cfg.species_maps)

    report = {
        "scenario": "ethanol_water_500kpa",
        "case_name": run_cfg.case_name,
        "pressure": run_cfg.pressure,
        "selected_bank_pressures": liquid_thermo.selected_bank_pressures.tolist(),
        "liquid_valid_temperature_range": list(liquid_thermo.valid_temperature_range()),
        "equilibrium": {
            "Ts": eq_result.Ts,
            "P": eq_result.P,
            "gamma_cond": eq_result.gamma_cond,
            "psat_cond": eq_result.psat_cond,
            "latent_cond": eq_result.latent_cond,
            "Xg_eq_full": eq_result.Xg_eq_full,
            "Yg_eq_full": eq_result.Yg_eq_full,
            "diagnostics": eq_result.diagnostics,
        },
        "state_pack_roundtrip": roundtrip,
        "bulk_props": {
            "diagnostics": props.diagnostics,
            "D_l_shape": list(props.D_l.shape) if props.D_l is not None else None,
            "D_g_shape": list(props.D_g.shape) if props.D_g is not None else None,
            "min_D_l": float(np.min(props.D_l)) if props.D_l is not None else None,
            "min_D_g": float(np.min(props.D_g)) if props.D_g is not None else None,
        },
        "geometry": {
            "n_liq_cells": mesh.n_liq,
            "n_gas_cells": mesh.n_gas,
            "region2_outer_face_index": metrics.region2_outer_face_index,
        },
    }

    report_path = _write_report("functional_ethanol_water_500kpa.json", report)

    assert report_path.exists()
    assert liquid_thermo.selected_bank_pressures.tolist() == [500000.0, 500000.0]
    assert np.isclose(np.sum(eq_result.Yg_eq_full), 1.0, rtol=0.0, atol=1.0e-12)
    assert np.all(eq_result.gamma_cond > 0.0)
    assert roundtrip["max_abs_Yl_error"] == 0.0
    assert props.diagnostics["all_finite"] is True
    assert props.D_l is not None
    assert props.D_l.shape == (mesh.n_liq, run_cfg.species_maps.n_liq_full)
