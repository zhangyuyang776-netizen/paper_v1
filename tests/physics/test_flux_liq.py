from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import cantera as ct
import numpy as np

from core.config_loader import load_and_validate_config
from core.grid import build_initial_grid
from core.preprocess import normalize_config
from core.types import GeometryState, InterfaceState, Mesh1D, RegionSlices, State
from physics.flux_liq import (
    build_liquid_center_boundary_flux,
    build_liquid_internal_diffusion_package,
    build_liquid_internal_energy_flux_package,
)
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
    return build_initial_grid(run_cfg, geometry)


def _make_stub_mesh() -> Mesh1D:
    return Mesh1D(
        r_faces=np.array([0.0, 1.0, 2.0, 3.0, 4.0], dtype=np.float64),
        r_centers=np.array([0.5, 1.5, 2.5, 3.5], dtype=np.float64),
        volumes=np.ones(4, dtype=np.float64),
        face_areas=np.array([0.0, 1.0, 2.0, 3.0, 4.0], dtype=np.float64),
        dr=np.ones(4, dtype=np.float64),
        region_slices=RegionSlices(
            liq=slice(0, 3),
            gas_near=slice(3, 4),
            gas_far=slice(4, 4),
            gas_all=slice(3, 4),
        ),
        interface_face_index=3,
        interface_cell_liq=2,
        interface_cell_gas=3,
    )


def _multicomponent_run_cfg():
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
    return _run_config_from_raw(raw_cfg)


def _liquid_closure_index(run_cfg) -> int | None:
    closure_name = run_cfg.species_maps.liq_closure_name
    if closure_name is None:
        return None
    return run_cfg.species_maps.liq_full_names.index(closure_name)


def _single_component_run_cfg():
    return _run_config_from_raw(_base_raw_config())


def _build_initial_state(run_cfg):
    liquid_thermo, gas_thermo = _build_models(run_cfg)
    mesh = _build_initial_mesh(run_cfg)
    bundle = build_initial_state_bundle(run_cfg, mesh, gas_thermo, liquid_thermo)
    return mesh, bundle.state0, liquid_thermo


def _clone_state_with_liquid_fields(state: State, *, Tl: np.ndarray | None = None, Yl_full: np.ndarray | None = None) -> State:
    return State(
        Tl=state.Tl.copy() if Tl is None else np.asarray(Tl, dtype=np.float64),
        Yl_full=state.Yl_full.copy() if Yl_full is None else np.asarray(Yl_full, dtype=np.float64),
        Tg=state.Tg.copy(),
        Yg_full=state.Yg_full.copy(),
        interface=InterfaceState(
            Ts=state.interface.Ts,
            mpp=state.interface.mpp,
            Ys_g_full=state.interface.Ys_g_full.copy(),
            Ys_l_full=state.interface.Ys_l_full.copy(),
        ),
        rho_l=state.rho_l.copy() if state.rho_l is not None else None,
        rho_g=state.rho_g.copy() if state.rho_g is not None else None,
        hl=state.hl.copy() if state.hl is not None else None,
        hg=state.hg.copy() if state.hg is not None else None,
        Xg_full=state.Xg_full.copy() if state.Xg_full is not None else None,
        time=state.time,
        state_id=state.state_id,
    )


def test_liquid_diffusion_zero_for_uniform_composition() -> None:
    run_cfg = _multicomponent_run_cfg()
    mesh, state, liquid_thermo = _build_initial_state(run_cfg)
    uniform_Y = np.tile(np.array([[0.7, 0.3]], dtype=np.float64), (mesh.n_liq, 1))
    uniform_state = _clone_state_with_liquid_fields(state, Yl_full=uniform_Y)

    diff_pkg = build_liquid_internal_diffusion_package(
        mesh,
        uniform_state,
        liquid_thermo,
        liquid_closure_index=_liquid_closure_index(run_cfg),
    )

    assert diff_pkg.face_indices.shape[0] == mesh.n_liq - 1
    assert np.allclose(diff_pkg.grad_Y_full, 0.0)
    assert np.allclose(diff_pkg.J_diff_full, 0.0)


def test_liquid_conduction_zero_for_uniform_temperature() -> None:
    run_cfg = _multicomponent_run_cfg()
    mesh, state, liquid_thermo = _build_initial_state(run_cfg)
    uniform_T = np.full(mesh.n_liq, 320.0, dtype=np.float64)
    uniform_state = _clone_state_with_liquid_fields(state, Tl=uniform_T)

    diff_pkg = build_liquid_internal_diffusion_package(
        mesh,
        uniform_state,
        liquid_thermo,
        liquid_closure_index=_liquid_closure_index(run_cfg),
    )
    energy_pkg = build_liquid_internal_energy_flux_package(mesh, uniform_state, liquid_thermo, diff_pkg)

    assert np.allclose(energy_pkg.grad_T, 0.0)
    assert np.allclose(energy_pkg.q_cond, 0.0)
    assert np.allclose(energy_pkg.q_total, energy_pkg.q_species_diff)


def test_single_component_liquid_naturally_degenerates_to_zero_species_diffusion() -> None:
    run_cfg = _single_component_run_cfg()
    mesh, state, liquid_thermo = _build_initial_state(run_cfg)

    diff_pkg = build_liquid_internal_diffusion_package(mesh, state, liquid_thermo)
    energy_pkg = build_liquid_internal_energy_flux_package(mesh, state, liquid_thermo, diff_pkg)

    assert diff_pkg.J_diff_full.shape[1] == 1
    assert np.allclose(diff_pkg.J_diff_full, 0.0)
    assert np.allclose(energy_pkg.q_species_diff, 0.0)
    assert np.allclose(energy_pkg.q_total, energy_pkg.q_cond)


def test_internal_face_package_excludes_center_and_interface_face() -> None:
    run_cfg = _multicomponent_run_cfg()
    mesh, state, liquid_thermo = _build_initial_state(run_cfg)

    diff_pkg = build_liquid_internal_diffusion_package(
        mesh,
        state,
        liquid_thermo,
        liquid_closure_index=_liquid_closure_index(run_cfg),
    )
    center_J, center_q = build_liquid_center_boundary_flux(state.n_liq_species_full)

    assert 0 not in diff_pkg.face_indices
    assert mesh.interface_face_index not in diff_pkg.face_indices
    assert np.array_equal(diff_pkg.face_indices, np.arange(1, mesh.n_liq, dtype=np.int64))
    assert np.allclose(center_J, 0.0)
    assert center_q == 0.0


def test_energy_flux_matches_cond_plus_species_enthalpy_diffusion_identity() -> None:
    run_cfg = _multicomponent_run_cfg()
    mesh, state, liquid_thermo = _build_initial_state(run_cfg)

    diff_pkg = build_liquid_internal_diffusion_package(
        mesh,
        state,
        liquid_thermo,
        liquid_closure_index=_liquid_closure_index(run_cfg),
    )
    energy_pkg = build_liquid_internal_energy_flux_package(mesh, state, liquid_thermo, diff_pkg)

    expected = energy_pkg.q_cond + np.sum(diff_pkg.J_diff_full * energy_pkg.h_face_full, axis=1)
    assert np.allclose(energy_pkg.q_species_diff, np.sum(diff_pkg.J_diff_full * energy_pkg.h_face_full, axis=1))
    assert np.allclose(energy_pkg.q_total, expected)


def test_multicomponent_liquid_diffusion_flux_sums_to_zero_per_face() -> None:
    run_cfg = _multicomponent_run_cfg()
    mesh, state, liquid_thermo = _build_initial_state(run_cfg)

    diff_pkg = build_liquid_internal_diffusion_package(
        mesh,
        state,
        liquid_thermo,
        liquid_closure_index=_liquid_closure_index(run_cfg),
    )

    assert np.allclose(np.sum(diff_pkg.J_diff_full, axis=1), 0.0, atol=1.0e-12, rtol=0.0)


class _StubLiquidProps:
    def __init__(self, n_species: int) -> None:
        self.n_species = n_species
        self.conductivity_calls = 0
        self.diffusivity_calls = 0
        self.pure_enthalpy_calls = 0

    def conductivity(self, T: float, Y_full: np.ndarray) -> float:
        self.conductivity_calls += 1
        return 2.0 + 0.001 * float(T) + 0.1 * float(np.sum(Y_full))

    def diffusivity(self, T: float, Y_full: np.ndarray) -> np.ndarray | None:
        self.diffusivity_calls += 1
        if self.n_species == 1:
            return None
        base = 1.0e-9 + 1.0e-12 * float(T)
        return np.full(self.n_species, base, dtype=np.float64)

    def pure_enthalpy_vector(self, T: float) -> np.ndarray:
        self.pure_enthalpy_calls += 1
        return np.linspace(1.0, float(self.n_species), self.n_species, dtype=np.float64) * (1000.0 + float(T))


def test_flux_liq_uses_liquid_properties_layer_for_D_k_h() -> None:
    mesh = _make_stub_mesh()
    state = State(
        Tl=np.array([300.0, 310.0, 320.0], dtype=np.float64),
        Yl_full=np.array([[0.7, 0.3], [0.6, 0.4], [0.5, 0.5]], dtype=np.float64),
        Tg=np.array([1000.0], dtype=np.float64),
        Yg_full=np.array([[0.0, 0.0, 0.21, 0.79]], dtype=np.float64),
        interface=InterfaceState(
            Ts=300.0,
            mpp=0.0,
            Ys_g_full=np.array([0.08, 0.02, 0.189, 0.711], dtype=np.float64),
            Ys_l_full=np.array([0.7, 0.3], dtype=np.float64),
        ),
        rho_l=np.array([900.0, 905.0, 910.0], dtype=np.float64),
        rho_g=np.array([1.0], dtype=np.float64),
        hl=np.array([1.0, 2.0, 3.0], dtype=np.float64),
        hg=np.array([10.0], dtype=np.float64),
        time=0.0,
        state_id="stub_flux_liq",
    )
    stub = _StubLiquidProps(n_species=2)

    diff_pkg = build_liquid_internal_diffusion_package(mesh, state, stub, liquid_closure_index=1)
    energy_pkg = build_liquid_internal_energy_flux_package(mesh, state, stub, diff_pkg)

    assert stub.diffusivity_calls == mesh.n_liq
    assert stub.conductivity_calls == mesh.n_liq
    assert stub.pure_enthalpy_calls == mesh.n_liq
    assert diff_pkg.D_face_full.shape == (mesh.n_liq - 1, 2)
    assert energy_pkg.k_face.shape == (mesh.n_liq - 1,)
