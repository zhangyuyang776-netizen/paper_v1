from __future__ import annotations

from copy import deepcopy
from math import erfc, sqrt
from pathlib import Path

import cantera as ct
import numpy as np

from core.config_loader import load_and_validate_config
from core.grid import build_initial_grid
from core.layout import build_layout
from core.preprocess import normalize_config
from core.state_pack import pack_state_to_array, unpack_array_to_state
from core.types import GeometryState
from physics.initial import build_initial_state_bundle
from properties.gas import build_gas_thermo_model
from properties.liquid import build_liquid_thermo_model
from properties.liquid_db import build_liquid_database


TESTS_DIR = Path(__file__).resolve().parents[1]
PAPER_ROOT = TESTS_DIR.parent
CASE_CONFIG_PATH = PAPER_ROOT / "cases" / "config.yaml"


def _base_raw_config() -> dict:
    return load_and_validate_config(CASE_CONFIG_PATH)


def _run_config_from_raw(raw_cfg: dict):
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
    return liquid_thermo, gas_thermo


def _build_initial_mesh(run_cfg):
    geometry = GeometryState(
        t=run_cfg.time_stepper.t0,
        dt=run_cfg.time_stepper.dt_start,
        a=run_cfg.mesh.a0,
        dot_a=0.0,
        r_end=run_cfg.mesh.r_end,
        step_index=0,
        outer_iter_index=0,
    )
    mesh = build_initial_grid(run_cfg, geometry)
    layout = build_layout(run_cfg, mesh)
    return geometry, mesh, layout


def _expected_multiplier(r: np.ndarray, *, a0: float, D: float, t0: float) -> np.ndarray:
    return (a0 / r) * np.array([erfc((float(rv) - a0) / (2.0 * sqrt(D * t0))) for rv in r], dtype=np.float64)


def test_far_field_thermal_diffusivity_uses_far_field_backend_props() -> None:
    run_cfg = _run_config_from_raw(_base_raw_config())
    liquid_thermo, gas_thermo = _build_models(run_cfg)
    _, mesh, _ = _build_initial_mesh(run_cfg)

    bundle = build_initial_state_bundle(run_cfg, mesh, gas_thermo, liquid_thermo)

    rho = gas_thermo.density_mass(
        run_cfg.initialization.gas_temperature,
        run_cfg.initialization.gas_y_full_0,
        run_cfg.initialization.gas_pressure,
    )
    cp = gas_thermo.cp_mass(
        run_cfg.initialization.gas_temperature,
        run_cfg.initialization.gas_y_full_0,
        run_cfg.initialization.gas_pressure,
    )
    k = gas_thermo.conductivity(
        run_cfg.initialization.gas_temperature,
        run_cfg.initialization.gas_y_full_0,
        run_cfg.initialization.gas_pressure,
    )
    expected_D = k / (cp * rho)

    assert np.isclose(bundle.info.rho_g_inf, rho)
    assert np.isclose(bundle.info.cp_g_inf, cp)
    assert np.isclose(bundle.info.k_g_inf, k)
    assert np.isclose(bundle.info.D_T_g_inf, expected_D)


def test_initial_state_bundle_single_case_matches_profiles_and_project_rules() -> None:
    run_cfg = _run_config_from_raw(_base_raw_config())
    liquid_thermo, gas_thermo = _build_models(run_cfg)
    _, mesh, layout = _build_initial_mesh(run_cfg)

    bundle = build_initial_state_bundle(run_cfg, mesh, gas_thermo, liquid_thermo, layout=layout)
    state0 = bundle.state0

    assert bundle.dot_a0 == 0.0
    assert state0.time == 0.0
    assert state0.interface.mpp == 0.0
    assert state0.interface.Ts == run_cfg.initialization.liquid_temperature
    assert np.allclose(state0.interface.Ys_l_full, run_cfg.initialization.liquid_y_full_0)
    assert np.isclose(np.sum(state0.interface.Ys_g_full), 1.0)
    assert np.all(state0.interface.Ys_g_full >= 0.0)

    r_gas = mesh.r_centers[mesh.gas_slice]
    multiplier = _expected_multiplier(
        r_gas,
        a0=run_cfg.a0,
        D=bundle.info.D_T_g_inf,
        t0=run_cfg.initialization.t_init_T,
    )
    expected_Tg = run_cfg.initialization.gas_temperature + multiplier * (
        run_cfg.initialization.liquid_temperature - run_cfg.initialization.gas_temperature
    )
    assert np.allclose(state0.Tg, expected_Tg)

    expected_Yg = run_cfg.initialization.gas_y_full_0[None, :] + multiplier[:, None] * (
        state0.interface.Ys_g_full[None, :] - run_cfg.initialization.gas_y_full_0[None, :]
    )
    assert np.allclose(state0.Yg_full, expected_Yg)
    assert np.allclose(np.sum(state0.Yg_full, axis=1), 1.0)

    assert np.allclose(state0.Tl, run_cfg.initialization.liquid_temperature)
    assert np.allclose(state0.Yl_full, run_cfg.initialization.liquid_y_full_0[None, :])
    assert state0.rho_l is not None and np.all(state0.rho_l > 0.0)
    assert state0.rho_g is not None and np.all(state0.rho_g > 0.0)
    assert state0.hl is not None and np.all(np.isfinite(state0.hl))
    assert state0.hg is not None and np.all(np.isfinite(state0.hg))

    vec = pack_state_to_array(state0, layout, run_cfg.species_maps)
    state_rt = unpack_array_to_state(vec, layout, run_cfg.species_maps, time=state0.time, state_id=state0.state_id)
    assert np.allclose(state_rt.Tl, state0.Tl)
    assert np.allclose(state_rt.Yl_full, state0.Yl_full)
    assert np.allclose(state_rt.Tg, state0.Tg)
    assert np.allclose(state_rt.Yg_full, state0.Yg_full)
    assert np.allclose(state_rt.interface.Ys_l_full, state0.interface.Ys_l_full)
    assert np.allclose(state_rt.interface.Ys_g_full, state0.interface.Ys_g_full)
    assert state_rt.interface.Ts == state0.interface.Ts
    assert state_rt.interface.mpp == state0.interface.mpp


def test_initial_state_bundle_multicomponent_keeps_full_order_and_seed_rule() -> None:
    raw_cfg = deepcopy(_base_raw_config())
    raw_cfg["initialization"]["gas_pressure"] = 5.0e5
    raw_cfg["species"]["liquid_closure_species"] = "water"
    raw_cfg["species"]["liquid_to_gas_species_map"] = {
        "ethanol": "C2H5OH",
        "water": "H2O",
    }
    raw_cfg["initialization"]["liquid_composition"] = {
        "ethanol": 0.7,
        "water": 0.3,
    }
    raw_cfg["initialization"]["Y_vap_if0"] = {
        "C2H5OH": 0.08,
        "H2O": 0.02,
    }
    run_cfg = _run_config_from_raw(raw_cfg)
    liquid_thermo, gas_thermo = _build_models(run_cfg)
    _, mesh, layout = _build_initial_mesh(run_cfg)

    bundle = build_initial_state_bundle(run_cfg, mesh, gas_thermo, liquid_thermo, layout=layout)
    state0 = bundle.state0

    assert state0.n_liq_species_full == 2
    assert state0.n_gas_species_full == 4
    assert np.allclose(state0.interface.Ys_l_full, [0.7, 0.3])
    assert np.isclose(np.sum(state0.interface.Ys_g_full), 1.0)

    gas_names = run_cfg.species_maps.gas_full_names
    ethanol_idx = gas_names.index("C2H5OH")
    water_idx = gas_names.index("H2O")
    o2_idx = gas_names.index("O2")
    n2_idx = gas_names.index("N2")
    assert np.isclose(state0.interface.Ys_g_full[ethanol_idx], 0.08)
    assert np.isclose(state0.interface.Ys_g_full[water_idx], 0.02)

    alpha = (1.0 - 0.10) / (1.0 - 0.0)
    assert np.isclose(state0.interface.Ys_g_full[o2_idx], alpha * run_cfg.initialization.gas_y_full_0[o2_idx])
    assert np.isclose(state0.interface.Ys_g_full[n2_idx], alpha * run_cfg.initialization.gas_y_full_0[n2_idx])

    assert state0.rho_l is not None and np.all(state0.rho_l > 0.0)
    assert state0.rho_g is not None and np.all(state0.rho_g > 0.0)
    assert state0.hl is not None and np.all(np.isfinite(state0.hl))
    assert state0.hg is not None and np.all(np.isfinite(state0.hg))
    assert np.allclose(np.sum(state0.Yg_full, axis=1), 1.0)


def test_initial_state_bundle_uses_species_maps_not_raw_species_mapping() -> None:
    raw_cfg = deepcopy(_base_raw_config())
    raw_cfg["initialization"]["gas_pressure"] = 5.0e5
    raw_cfg["species"]["liquid_closure_species"] = "water"
    raw_cfg["species"]["liquid_to_gas_species_map"] = {
        "ethanol": "C2H5OH",
        "water": "H2O",
    }
    raw_cfg["initialization"]["liquid_composition"] = {
        "ethanol": 0.7,
        "water": 0.3,
    }
    raw_cfg["initialization"]["Y_vap_if0"] = {
        "C2H5OH": 0.08,
        "H2O": 0.02,
    }
    run_cfg = _run_config_from_raw(raw_cfg)
    liquid_thermo, gas_thermo = _build_models(run_cfg)
    _, mesh, layout = _build_initial_mesh(run_cfg)

    original_species_cfg = run_cfg.species
    object.__setattr__(
        run_cfg,
        "species",
        type(original_species_cfg)(
            gas_closure_species=original_species_cfg.gas_closure_species,
            liquid_closure_species=original_species_cfg.liquid_closure_species,
            liquid_to_gas_species_map={"ethanol": "H2O", "water": "C2H5OH"},
        ),
    )
    try:
        bundle = build_initial_state_bundle(run_cfg, mesh, gas_thermo, liquid_thermo, layout=layout)
    finally:
        object.__setattr__(run_cfg, "species", original_species_cfg)

    gas_names = run_cfg.species_maps.gas_full_names
    ethanol_idx = gas_names.index("C2H5OH")
    water_idx = gas_names.index("H2O")
    assert np.isclose(bundle.state0.interface.Ys_g_full[ethanol_idx], 0.08)
    assert np.isclose(bundle.state0.interface.Ys_g_full[water_idx], 0.02)


def test_initial_state_bundle_rejects_invalid_interface_seed_sum() -> None:
    raw_cfg = deepcopy(_base_raw_config())
    raw_cfg["initialization"]["Y_vap_if0"] = {"C2H5OH": 1.0}
    run_cfg = _run_config_from_raw(raw_cfg)
    liquid_thermo, gas_thermo = _build_models(run_cfg)
    _, mesh, layout = _build_initial_mesh(run_cfg)

    try:
        build_initial_state_bundle(run_cfg, mesh, gas_thermo, liquid_thermo, layout=layout)
    except Exception as exc:  # explicit type is enough for behavior lock
        assert "strictly less than 1" in str(exc)
    else:
        raise AssertionError("expected invalid interface seed sum to be rejected")
