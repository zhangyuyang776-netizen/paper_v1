from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from core.preprocess import PreprocessError, normalize_config


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
            "T_min_l": 200.0,
            "T_max_l": 800.0,
            "T_min_g": 200.0,
            "T_max_g": 4000.0,
            "liq_h_inv_tol": 1.0e-10,
            "liq_h_inv_max_iter": 50,
            "gas_h_inv_tol": 1.0e-10,
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


def _prepare_case_files(tmp_path: Path) -> tuple[Path, Path, Path]:
    case_dir = tmp_path / "case"
    data_dir = case_dir / "data"
    mech_dir = case_dir / "mech"
    data_dir.mkdir(parents=True)
    mech_dir.mkdir(parents=True)
    (data_dir / "liquid_db.yaml").write_text("species: {}\n", encoding="utf-8")
    (mech_dir / "gas.yaml").write_text("dummy: true\n", encoding="utf-8")
    config_path = case_dir / "case.yaml"
    config_path.write_text("case: {}\n", encoding="utf-8")
    return case_dir, data_dir / "liquid_db.yaml", mech_dir / "gas.yaml"


def _patch_gas_species(monkeypatch: pytest.MonkeyPatch, species_names: tuple[str, ...]) -> None:
    monkeypatch.setattr(
        "core.preprocess._load_gas_species_names",
        lambda mechanism_path, phase_name="gas": species_names,
    )


def test_preprocess_resolves_relative_paths_against_config_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case_dir, liquid_db_path, gas_mech_path = _prepare_case_files(tmp_path)
    _patch_gas_species(monkeypatch, ("N2", "O2", "C2H5OH"))
    raw_cfg = make_minimal_valid_raw_config()
    config_path = case_dir / "case.yaml"

    run_cfg = normalize_config(raw_cfg, source_path=config_path)

    assert run_cfg.paths.case_root == case_dir.resolve()
    assert run_cfg.paths.liquid_database_path == liquid_db_path.resolve()
    assert run_cfg.paths.gas_mechanism_path == gas_mech_path.resolve()
    assert run_cfg.paths.output_root == (case_dir / "out").resolve()


def test_preprocess_keeps_absolute_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    case_dir, liquid_db_path, gas_mech_path = _prepare_case_files(tmp_path)
    _patch_gas_species(monkeypatch, ("N2", "O2", "C2H5OH"))
    raw_cfg = make_minimal_valid_raw_config()
    raw_cfg["paths"]["liquid_database_path"] = str(liquid_db_path.resolve())
    raw_cfg["paths"]["gas_mechanism_path"] = str(gas_mech_path.resolve())
    raw_cfg["paths"]["output_root"] = str((tmp_path / "abs_out").resolve())

    run_cfg = normalize_config(raw_cfg, source_path=case_dir / "case.yaml")

    assert run_cfg.paths.liquid_database_path == liquid_db_path.resolve()
    assert run_cfg.paths.gas_mechanism_path == gas_mech_path.resolve()
    assert run_cfg.paths.output_root == (tmp_path / "abs_out").resolve()


def test_preprocess_missing_external_file_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    case_dir, _, _ = _prepare_case_files(tmp_path)
    _patch_gas_species(monkeypatch, ("N2", "O2", "C2H5OH"))
    raw_cfg = make_minimal_valid_raw_config()
    raw_cfg["paths"]["gas_mechanism_path"] = "mech/missing.yaml"

    with pytest.raises(PreprocessError, match="gas_mechanism_path does not exist"):
        normalize_config(raw_cfg, source_path=case_dir / "case.yaml")


def test_preprocess_rejects_gas_closure_not_in_mechanism(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case_dir, _, _ = _prepare_case_files(tmp_path)
    _patch_gas_species(monkeypatch, ("O2", "C2H5OH"))
    raw_cfg = make_minimal_valid_raw_config()

    with pytest.raises(PreprocessError, match="gas_closure_species"):
        normalize_config(raw_cfg, source_path=case_dir / "case.yaml")


def test_preprocess_rejects_unknown_gas_initial_species(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case_dir, _, _ = _prepare_case_files(tmp_path)
    _patch_gas_species(monkeypatch, ("N2", "O2", "C2H5OH"))
    raw_cfg = make_minimal_valid_raw_config()
    raw_cfg["initialization"]["gas_composition"] = {"O2": 0.2, "AR": 0.8}

    with pytest.raises(PreprocessError, match="unknown gas species"):
        normalize_config(raw_cfg, source_path=case_dir / "case.yaml")


def test_preprocess_single_component_liquid_normalizes_to_no_reduced_species(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case_dir, _, _ = _prepare_case_files(tmp_path)
    _patch_gas_species(monkeypatch, ("N2", "O2", "C2H5OH"))
    raw_cfg = make_minimal_valid_raw_config()
    raw_cfg["species"]["liquid_closure_species"] = "ethanol"

    run_cfg = normalize_config(raw_cfg, source_path=case_dir / "case.yaml")

    assert run_cfg.species_maps.liq_active_names == ()
    assert run_cfg.species_maps.liq_closure_name is None
    assert run_cfg.species_maps.liq_reduced_to_full.size == 0


def test_preprocess_derives_single_component_unknowns_profile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case_dir, _, _ = _prepare_case_files(tmp_path)
    _patch_gas_species(monkeypatch, ("N2", "O2", "C2H5OH"))
    raw_cfg = make_minimal_valid_raw_config()

    run_cfg = normalize_config(raw_cfg, source_path=case_dir / "case.yaml")

    assert run_cfg.unknowns_profile == "U_A"


def test_preprocess_multicomponent_liquid_requires_valid_closure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case_dir, _, _ = _prepare_case_files(tmp_path)
    _patch_gas_species(monkeypatch, ("N2", "O2", "A_g", "B_g"))
    raw_cfg = make_minimal_valid_raw_config()
    raw_cfg["initialization"]["liquid_composition"] = {"A": 0.7, "B": 0.3}
    raw_cfg["species"]["liquid_closure_species"] = None
    raw_cfg["species"]["liquid_to_gas_species_map"] = {"A": "A_g", "B": "B_g"}
    raw_cfg["initialization"]["Y_vap_if0"] = {"A_g": 0.1}

    with pytest.raises(PreprocessError, match="multicomponent liquid requires liquid_closure_species"):
        normalize_config(raw_cfg, source_path=case_dir / "case.yaml")


def test_preprocess_rejects_liquid_closure_outside_liquid_species(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case_dir, _, _ = _prepare_case_files(tmp_path)
    _patch_gas_species(monkeypatch, ("N2", "O2", "A_g", "B_g"))
    raw_cfg = make_minimal_valid_raw_config()
    raw_cfg["initialization"]["liquid_composition"] = {"A": 0.7, "B": 0.3}
    raw_cfg["species"]["liquid_closure_species"] = "C"
    raw_cfg["species"]["liquid_to_gas_species_map"] = {"A": "A_g", "B": "B_g"}
    raw_cfg["initialization"]["Y_vap_if0"] = {"A_g": 0.1}

    with pytest.raises(PreprocessError, match="liquid_closure_species"):
        normalize_config(raw_cfg, source_path=case_dir / "case.yaml")


def test_preprocess_rejects_liquid_to_gas_target_not_in_gas_species(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case_dir, _, _ = _prepare_case_files(tmp_path)
    _patch_gas_species(monkeypatch, ("N2", "O2", "A_g"))
    raw_cfg = make_minimal_valid_raw_config()
    raw_cfg["species"]["liquid_to_gas_species_map"] = {"ethanol": "MISSING"}
    raw_cfg["initialization"]["Y_vap_if0"] = {"MISSING": 0.1}

    with pytest.raises(PreprocessError, match="gas targets not present in mechanism"):
        normalize_config(raw_cfg, source_path=case_dir / "case.yaml")


def test_preprocess_rejects_non_unique_liquid_to_gas_mapping(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case_dir, _, _ = _prepare_case_files(tmp_path)
    _patch_gas_species(monkeypatch, ("N2", "O2", "A_g"))
    raw_cfg = make_minimal_valid_raw_config()
    raw_cfg["initialization"]["liquid_composition"] = {"A": 0.5, "B": 0.5}
    raw_cfg["species"]["liquid_closure_species"] = "B"
    raw_cfg["species"]["liquid_to_gas_species_map"] = {"A": "A_g", "B": "A_g"}
    raw_cfg["initialization"]["Y_vap_if0"] = {"A_g": 0.1}

    with pytest.raises(PreprocessError, match="one-to-one"):
        normalize_config(raw_cfg, source_path=case_dir / "case.yaml")


def test_preprocess_builds_liquid_to_gas_full_index_map(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case_dir, _, _ = _prepare_case_files(tmp_path)
    _patch_gas_species(monkeypatch, ("N2", "A_g", "B_g", "O2"))
    raw_cfg = make_minimal_valid_raw_config()
    raw_cfg["initialization"]["liquid_composition"] = {"A": 0.7, "B": 0.3}
    raw_cfg["species"]["liquid_closure_species"] = "B"
    raw_cfg["species"]["liquid_to_gas_species_map"] = {"A": "A_g", "B": "B_g"}
    raw_cfg["initialization"]["Y_vap_if0"] = {"A_g": 0.1, "B_g": 0.05}

    run_cfg = normalize_config(raw_cfg, source_path=case_dir / "case.yaml")

    np.testing.assert_array_equal(run_cfg.species_maps.liq_full_to_gas_full, np.array([1, 2], dtype=np.int64))


def test_preprocess_builds_gas_full_initial_vector(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case_dir, _, _ = _prepare_case_files(tmp_path)
    _patch_gas_species(monkeypatch, ("N2", "O2", "C2H5OH"))
    raw_cfg = make_minimal_valid_raw_config()

    run_cfg = normalize_config(raw_cfg, source_path=case_dir / "case.yaml")

    np.testing.assert_allclose(run_cfg.initialization.gas_y_full_0, np.array([0.79, 0.21, 0.0]))


def test_preprocess_builds_liquid_full_initial_vector(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case_dir, _, _ = _prepare_case_files(tmp_path)
    _patch_gas_species(monkeypatch, ("N2", "O2", "A_g", "B_g"))
    raw_cfg = make_minimal_valid_raw_config()
    raw_cfg["initialization"]["liquid_composition"] = {"A": 0.7, "B": 0.3}
    raw_cfg["species"]["liquid_closure_species"] = "B"
    raw_cfg["species"]["liquid_to_gas_species_map"] = {"A": "A_g", "B": "B_g"}
    raw_cfg["initialization"]["Y_vap_if0"] = {"A_g": 0.1}

    run_cfg = normalize_config(raw_cfg, source_path=case_dir / "case.yaml")

    np.testing.assert_allclose(run_cfg.initialization.liquid_y_full_0, np.array([0.7, 0.3]))


def test_preprocess_builds_interface_vapor_full_vector(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case_dir, _, _ = _prepare_case_files(tmp_path)
    _patch_gas_species(monkeypatch, ("N2", "O2", "C2H5OH"))
    raw_cfg = make_minimal_valid_raw_config()

    run_cfg = normalize_config(raw_cfg, source_path=case_dir / "case.yaml")

    np.testing.assert_allclose(run_cfg.initialization.y_vap_if0_gas_full, np.array([0.0, 0.0, 0.1]))


def test_preprocess_rejects_y_vap_if0_outside_mapped_vapor_species(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case_dir, _, _ = _prepare_case_files(tmp_path)
    _patch_gas_species(monkeypatch, ("N2", "O2", "C2H5OH"))
    raw_cfg = make_minimal_valid_raw_config()
    raw_cfg["initialization"]["Y_vap_if0"] = {"O2": 0.1}

    with pytest.raises(PreprocessError, match="outside mapped vapor set"):
        normalize_config(raw_cfg, source_path=case_dir / "case.yaml")


def test_preprocess_builds_gas_reduced_mappings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case_dir, _, _ = _prepare_case_files(tmp_path)
    _patch_gas_species(monkeypatch, ("N2", "O2", "C2H5OH"))
    raw_cfg = make_minimal_valid_raw_config()

    run_cfg = normalize_config(raw_cfg, source_path=case_dir / "case.yaml")

    assert run_cfg.species_maps.gas_active_names == ("O2", "C2H5OH")
    np.testing.assert_array_equal(run_cfg.species_maps.gas_full_to_reduced, np.array([-1, 0, 1], dtype=np.int64))


def test_preprocess_builds_liquid_reduced_mappings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case_dir, _, _ = _prepare_case_files(tmp_path)
    _patch_gas_species(monkeypatch, ("N2", "O2", "A_g", "B_g"))
    raw_cfg = make_minimal_valid_raw_config()
    raw_cfg["initialization"]["liquid_composition"] = {"A": 0.7, "B": 0.3}
    raw_cfg["species"]["liquid_closure_species"] = "B"
    raw_cfg["species"]["liquid_to_gas_species_map"] = {"A": "A_g", "B": "B_g"}
    raw_cfg["initialization"]["Y_vap_if0"] = {"A_g": 0.1}

    run_cfg = normalize_config(raw_cfg, source_path=case_dir / "case.yaml")

    assert run_cfg.species_maps.liq_active_names == ("A",)
    np.testing.assert_array_equal(run_cfg.species_maps.liq_full_to_reduced, np.array([0, -1], dtype=np.int64))


def test_preprocess_derives_multicomponent_unknowns_profile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case_dir, _, _ = _prepare_case_files(tmp_path)
    _patch_gas_species(monkeypatch, ("N2", "O2", "A_g", "B_g"))
    raw_cfg = make_minimal_valid_raw_config()
    raw_cfg["initialization"]["liquid_composition"] = {"A": 0.7, "B": 0.3}
    raw_cfg["species"]["liquid_closure_species"] = "B"
    raw_cfg["species"]["liquid_to_gas_species_map"] = {"A": "A_g", "B": "B_g"}
    raw_cfg["initialization"]["Y_vap_if0"] = {"A_g": 0.1}

    run_cfg = normalize_config(raw_cfg, source_path=case_dir / "case.yaml")

    assert run_cfg.unknowns_profile == "U_B"


def test_preprocess_normalize_config_returns_run_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case_dir, _, _ = _prepare_case_files(tmp_path)
    _patch_gas_species(monkeypatch, ("N2", "O2", "C2H5OH"))
    raw_cfg = make_minimal_valid_raw_config()

    run_cfg = normalize_config(raw_cfg, source_path=case_dir / "case.yaml")

    assert run_cfg.case_name == "baseline_case"
    assert run_cfg.mesh.a0 == raw_cfg["mesh"]["a0"]


def test_preprocess_run_config_contains_resolved_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case_dir, liquid_db_path, gas_mech_path = _prepare_case_files(tmp_path)
    _patch_gas_species(monkeypatch, ("N2", "O2", "C2H5OH"))
    raw_cfg = make_minimal_valid_raw_config()

    run_cfg = normalize_config(raw_cfg, source_path=case_dir / "case.yaml")

    assert run_cfg.paths.config_path == (case_dir / "case.yaml").resolve()
    assert run_cfg.paths.liquid_database_path == liquid_db_path.resolve()
    assert run_cfg.paths.gas_mechanism_path == gas_mech_path.resolve()


def test_preprocess_injects_internal_defaults(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case_dir, _, _ = _prepare_case_files(tmp_path)
    _patch_gas_species(monkeypatch, ("N2", "O2", "C2H5OH"))
    raw_cfg = make_minimal_valid_raw_config()

    run_cfg = normalize_config(raw_cfg, source_path=case_dir / "case.yaml")

    assert run_cfg.gas_phase_name == "gas"
    assert run_cfg.liquid_model_name == "default"
    assert run_cfg.equilibrium_model_name == "default"
