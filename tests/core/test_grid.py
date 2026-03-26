from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from core.grid import REGION2_OUTER_RADIUS_FACTOR, build_grid_and_metrics, build_initial_grid, rebuild_grid
from core.preprocess import normalize_config
from core.types import GeometryState


def make_minimal_valid_raw_config() -> dict:
    return {
        "case": {
            "name": "baseline_case",
            "description": "transport-only baseline",
        },
        "paths": {
            "liquid_database_path": "data/liquid_db.yaml",
            "gas_mechanism_path": "mech/gas.yaml",
            "output_root": "out/",
        },
        "mesh": {
            "a0": 1.0e-4,
            "r_end": 1.0e-2,
            "n_liq": 40,
            "n_gas_near": 80,
            "far_stretch_ratio": 1.03,
        },
        "species": {
            "gas_closure_species": "N2",
            "liquid_closure_species": None,
            "liquid_to_gas_species_map": {
                "ethanol": "C2H5OH",
            },
        },
        "initialization": {
            "gas_temperature": 1000.0,
            "gas_pressure": 101325.0,
            "liquid_temperature": 300.0,
            "gas_composition": {
                "O2": 0.21,
                "N2": 0.79,
            },
            "liquid_composition": {
                "ethanol": 1.0,
            },
            "t_init_T": 1.0e-5,
            "Y_vap_if0": {
                "C2H5OH": 0.10,
            },
        },
        "time_stepper": {
            "t0": 0.0,
            "t_end": 1.0e-3,
            "dt_start": 1.0e-6,
            "dt_min": 1.0e-9,
            "dt_max": 1.0e-4,
            "retry_max_per_step": 4,
            "q_success_for_growth": 3,
            "growth_factor": 1.2,
            "shrink_factor": 0.5,
        },
        "outer_stepper": {
            "outer_max_iter": 8,
            "eps_dot_a_tol": 1.0e-6,
            "corrector_relaxation": 1.0,
        },
        "solver_inner_petsc": {
            "inner_max_iter": 50,
            "snes_rtol": 1.0e-8,
            "snes_atol": 1.0e-10,
            "snes_stol": 1.0e-12,
            "ksp_rtol": 1.0e-8,
            "pc_type": "ilu",
            "ksp_type": "gmres",
            "use_fieldsplit": False,
        },
        "recovery": {
            "rho_min": 1.0e-12,
            "m_min": 1.0e-20,
            "species_recovery_eps_abs": 1.0e-14,
            "Y_sum_tol": 1.0e-10,
            "Y_hard_tol": 1.0e-6,
            "h_abs_tol": 1.0e-8,
            "h_rel_tol": 1.0e-10,
            "h_check_tol": 1.0e-8,
            "T_step_tol": 1.0e-8,
            "T_min_l": 200.0,
            "T_max_l": 800.0,
            "T_min_g": 200.0,
            "T_max_g": 4000.0,
            "liquid_h_inv_max_iter": 50,
            "cp_min": 1.0,
            "gas_h_inv_max_iter": 50,
            "use_cantera_hpy_first": True,
        },
        "diagnostics": {
            "verbose_interface_panel": False,
            "verbose_property_warnings": True,
            "write_step_diag": True,
            "write_interface_diag": True,
            "write_failure_report": True,
            "output_every_n_steps": 10,
        },
        "output": {
            "write_spatial_fields": True,
            "write_spatial_species": False,
            "write_time_series_scalars": True,
            "write_time_series_species": False,
            "snapshot_format": "npz",
        },
        "validation": {
            "enable_mass_balance_check": True,
            "enable_energy_balance_check": True,
            "enable_state_bounds_check": True,
        },
    }


def _prepare_case_files(tmp_path: Path) -> Path:
    case_dir = tmp_path / "case"
    data_dir = case_dir / "data"
    mech_dir = case_dir / "mech"
    data_dir.mkdir(parents=True)
    mech_dir.mkdir(parents=True)
    (data_dir / "liquid_db.yaml").write_text("species: {}\n", encoding="utf-8")
    (mech_dir / "gas.yaml").write_text("dummy: true\n", encoding="utf-8")
    config_path = case_dir / "case.yaml"
    config_path.write_text("case: {}\n", encoding="utf-8")
    return config_path


def _patch_gas_species(monkeypatch: pytest.MonkeyPatch, species_names: tuple[str, ...]) -> None:
    monkeypatch.setattr(
        "core.preprocess._load_gas_species_names",
        lambda mechanism_path, phase_name="gas": species_names,
    )


def _make_run_cfg(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, raw_cfg: dict | None = None):
    config_path = _prepare_case_files(tmp_path)
    _patch_gas_species(monkeypatch, ("N2", "O2", "C2H5OH"))
    raw_cfg = make_minimal_valid_raw_config() if raw_cfg is None else raw_cfg
    return normalize_config(raw_cfg, source_path=config_path)


def _make_geometry(run_cfg, *, a: float | None = None, dot_a: float = -0.25) -> GeometryState:
    current_a = run_cfg.mesh.a0 if a is None else a
    return GeometryState(
        t=0.0,
        dt=1.0e-6,
        a=current_a,
        dot_a=dot_a,
        r_end=run_cfg.mesh.r_end,
        step_index=0,
        outer_iter_index=0,
    )


def test_build_initial_grid_constructs_three_regions(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run_cfg = _make_run_cfg(tmp_path, monkeypatch)
    geometry = _make_geometry(run_cfg)

    mesh = build_initial_grid(run_cfg, geometry)

    assert mesh.n_liq == run_cfg.mesh.n_liq
    assert mesh.n_gas >= run_cfg.mesh.n_gas_near + 1
    assert mesh.region_slices.gas_far.stop - mesh.region_slices.gas_far.start >= 1


def test_region1_last_face_equals_current_radius(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run_cfg = _make_run_cfg(tmp_path, monkeypatch)
    geometry = _make_geometry(run_cfg, a=8.0e-5)

    mesh = build_initial_grid(run_cfg, geometry)

    assert mesh.r_faces[mesh.interface_face_index] == pytest.approx(geometry.a)


def test_region2_outer_face_equals_five_a0(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run_cfg = _make_run_cfg(tmp_path, monkeypatch)
    geometry = _make_geometry(run_cfg, a=8.0e-5)

    mesh, metrics = build_grid_and_metrics(run_cfg, geometry)

    assert mesh.r_faces[metrics.region2_outer_face_index] == pytest.approx(
        REGION2_OUTER_RADIUS_FACTOR * run_cfg.mesh.a0
    )


def test_last_face_equals_r_end(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run_cfg = _make_run_cfg(tmp_path, monkeypatch)
    geometry = _make_geometry(run_cfg)

    mesh = build_initial_grid(run_cfg, geometry)

    assert mesh.r_faces[-1] == pytest.approx(run_cfg.mesh.r_end)


def test_region1_uniform_spacing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run_cfg = _make_run_cfg(tmp_path, monkeypatch)
    geometry = _make_geometry(run_cfg, a=8.0e-5)

    mesh = build_initial_grid(run_cfg, geometry)
    dr_region1 = mesh.dr[: run_cfg.mesh.n_liq]

    assert np.allclose(dr_region1, geometry.a / run_cfg.mesh.n_liq)


def test_region2_uniform_spacing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run_cfg = _make_run_cfg(tmp_path, monkeypatch)
    geometry = _make_geometry(run_cfg, a=8.0e-5)

    mesh = build_initial_grid(run_cfg, geometry)
    dr_region2 = mesh.dr[mesh.region_slices.gas_near]
    r_I = REGION2_OUTER_RADIUS_FACTOR * run_cfg.mesh.a0

    assert np.allclose(dr_region2, (r_I - geometry.a) / run_cfg.mesh.n_gas_near)


def test_region3_first_cell_matches_upper_envelope_rule(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_cfg = _make_run_cfg(tmp_path, monkeypatch)
    geometry = _make_geometry(run_cfg)

    mesh = build_initial_grid(run_cfg, geometry)
    dr_region3 = mesh.dr[mesh.region_slices.gas_far]

    assert dr_region3[0] == pytest.approx(
        REGION2_OUTER_RADIUS_FACTOR * run_cfg.mesh.a0 / run_cfg.mesh.n_gas_near
    )


def test_region3_spacing_is_monotone_non_decreasing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run_cfg = _make_run_cfg(tmp_path, monkeypatch)
    geometry = _make_geometry(run_cfg)

    mesh = build_initial_grid(run_cfg, geometry)
    dr_region3 = mesh.dr[mesh.region_slices.gas_far]

    if dr_region3.size > 2:
        assert np.all(np.diff(dr_region3[:-1]) >= -1.0e-15)


def test_region3_geometric_growth_before_last_cell(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run_cfg = _make_run_cfg(tmp_path, monkeypatch)
    geometry = _make_geometry(run_cfg)

    mesh = build_initial_grid(run_cfg, geometry)
    dr_region3 = mesh.dr[mesh.region_slices.gas_far]

    if dr_region3.size >= 3:
        assert np.allclose(
            dr_region3[1:-1],
            dr_region3[:-2] * run_cfg.mesh.far_stretch_ratio,
            rtol=1.0e-12,
            atol=1.0e-15,
        )


def test_rebuild_grid_region1_scales_with_current_a(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run_cfg = _make_run_cfg(tmp_path, monkeypatch)
    geometry_1 = _make_geometry(run_cfg, a=run_cfg.mesh.a0)
    geometry_2 = _make_geometry(run_cfg, a=0.5 * run_cfg.mesh.a0)

    mesh_1 = rebuild_grid(run_cfg, geometry_1)
    mesh_2 = rebuild_grid(run_cfg, geometry_2)

    region1_faces_1 = mesh_1.r_faces[: run_cfg.mesh.n_liq + 1] / geometry_1.a
    region1_faces_2 = mesh_2.r_faces[: run_cfg.mesh.n_liq + 1] / geometry_2.a

    assert np.allclose(region1_faces_1, region1_faces_2)


def test_rebuild_grid_region2_outer_boundary_stays_fixed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_cfg = _make_run_cfg(tmp_path, monkeypatch)
    geometry_1 = _make_geometry(run_cfg, a=run_cfg.mesh.a0)
    geometry_2 = _make_geometry(run_cfg, a=0.5 * run_cfg.mesh.a0)

    mesh_1, metrics_1 = build_grid_and_metrics(run_cfg, geometry_1)
    mesh_2, metrics_2 = build_grid_and_metrics(run_cfg, geometry_2)

    expected = REGION2_OUTER_RADIUS_FACTOR * run_cfg.mesh.a0
    assert mesh_1.r_faces[metrics_1.region2_outer_face_index] == pytest.approx(expected)
    assert mesh_2.r_faces[metrics_2.region2_outer_face_index] == pytest.approx(expected)


def test_rebuild_grid_region3_faces_are_time_invariant(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_cfg = _make_run_cfg(tmp_path, monkeypatch)
    geometry_1 = _make_geometry(run_cfg, a=run_cfg.mesh.a0)
    geometry_2 = _make_geometry(run_cfg, a=0.5 * run_cfg.mesh.a0)

    mesh_1, metrics_1 = build_grid_and_metrics(run_cfg, geometry_1)
    mesh_2, metrics_2 = build_grid_and_metrics(run_cfg, geometry_2)

    assert np.array_equal(
        mesh_1.r_faces[metrics_1.region2_outer_face_index :],
        mesh_2.r_faces[metrics_2.region2_outer_face_index :],
    )


def test_control_surface_velocity_piecewise_definition(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_cfg = _make_run_cfg(tmp_path, monkeypatch)
    geometry = _make_geometry(run_cfg, a=8.0e-5, dot_a=-0.15)

    mesh, metrics = build_grid_and_metrics(run_cfg, geometry)
    v_c_faces = metrics.v_c_faces

    assert v_c_faces[0] == pytest.approx(0.0)
    assert v_c_faces[metrics.interface_face_index] == pytest.approx(geometry.dot_a)
    assert v_c_faces[metrics.region2_outer_face_index] == pytest.approx(0.0)
    assert np.allclose(v_c_faces[metrics.region2_outer_face_index :], 0.0)


def test_grid_rejects_r_end_not_exceeding_region2_outer_radius(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_cfg = _make_run_cfg(tmp_path, monkeypatch)
    object.__setattr__(run_cfg.mesh, "r_end", REGION2_OUTER_RADIUS_FACTOR * run_cfg.mesh.a0)
    geometry = _make_geometry(run_cfg)

    with pytest.raises(ValueError, match="greater than 5 \\* a0"):
        build_initial_grid(run_cfg, geometry)


def test_grid_rejects_far_stretch_ratio_below_one(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run_cfg = _make_run_cfg(tmp_path, monkeypatch)
    object.__setattr__(run_cfg.mesh, "far_stretch_ratio", 0.99)
    geometry = _make_geometry(run_cfg)

    with pytest.raises(ValueError, match="far_stretch_ratio"):
        build_initial_grid(run_cfg, geometry)
