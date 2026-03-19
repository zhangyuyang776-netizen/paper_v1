from __future__ import annotations

from pathlib import Path

import pytest
import yaml

import core.config_loader as config_loader
from core.config_loader import (
    ConfigLoadError,
    load_and_validate_config,
    load_raw_config,
    validate_loaded_config,
)
from core.config_schema import ConfigSchemaError, ConfigValidationError


def write_text(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def write_yaml(path: Path, data: dict) -> Path:
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path


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


def test_load_raw_config_missing_file_raises(tmp_path: Path) -> None:
    cfg_path = tmp_path / "missing.yaml"
    with pytest.raises(ConfigLoadError, match="does not exist"):
        load_raw_config(cfg_path)


def test_load_raw_config_directory_path_raises(tmp_path: Path) -> None:
    with pytest.raises(ConfigLoadError, match="not a file"):
        load_raw_config(tmp_path)


def test_invalid_path_object_raises_config_load_error() -> None:
    with pytest.raises(ConfigLoadError, match="Invalid config path object"):
        load_raw_config(123)  # type: ignore[arg-type]


def test_empty_yaml_file_raises(tmp_path: Path) -> None:
    cfg_path = write_text(tmp_path / "empty.yaml", "")
    with pytest.raises(ConfigLoadError, match="empty|no YAML document"):
        load_raw_config(cfg_path)


def test_top_level_list_yaml_raises(tmp_path: Path) -> None:
    cfg_path = write_text(tmp_path / "bad.yaml", "- a\n- b\n")
    with pytest.raises(ConfigLoadError, match="Top-level YAML document must be a mapping"):
        load_raw_config(cfg_path)


def test_yaml_syntax_error_raises(tmp_path: Path) -> None:
    cfg_path = write_text(tmp_path / "bad.yaml", "case:\n  name: [unclosed\n")
    with pytest.raises(ConfigLoadError, match="Failed to parse YAML config"):
        load_raw_config(cfg_path)


def test_duplicate_top_level_key_raises(tmp_path: Path) -> None:
    text = """
case:
  name: a
case:
  name: b
"""
    cfg_path = write_text(tmp_path / "dup.yaml", text)
    with pytest.raises(ConfigLoadError, match="Duplicate key 'case'"):
        load_raw_config(cfg_path)


def test_duplicate_nested_key_raises(tmp_path: Path) -> None:
    text = """
mesh:
  a0: 1.0e-4
  a0: 2.0e-4
  r_end: 1.0e-2
  n_liq: 10
  n_gas_near: 20
  far_stretch_ratio: 1.03
"""
    cfg_path = write_text(tmp_path / "dup_nested.yaml", text)
    with pytest.raises(ConfigLoadError, match="Duplicate key 'a0'"):
        load_raw_config(cfg_path)


def test_load_raw_config_returns_raw_dict_even_if_schema_invalid(tmp_path: Path) -> None:
    cfg = {"case": {"name": "only_case"}}
    cfg_path = write_yaml(tmp_path / "partial.yaml", cfg)

    raw = load_raw_config(cfg_path)
    assert raw["case"]["name"] == "only_case"


def test_load_and_validate_config_runs_schema_validation(tmp_path: Path) -> None:
    cfg = {"case": {"name": "only_case"}}
    cfg_path = write_yaml(tmp_path / "partial.yaml", cfg)

    with pytest.raises(ConfigValidationError, match="Missing required section"):
        load_and_validate_config(cfg_path)


def test_load_and_validate_config_passes_for_valid_case(tmp_path: Path) -> None:
    cfg = make_minimal_valid_raw_config()
    cfg_path = write_yaml(tmp_path / "valid.yaml", cfg)

    loaded = load_and_validate_config(cfg_path)

    assert isinstance(loaded, dict)
    assert loaded["case"]["name"] == cfg["case"]["name"]
    assert loaded["mesh"]["a0"] == cfg["mesh"]["a0"]


def test_schema_error_includes_source_path(tmp_path: Path) -> None:
    cfg = make_minimal_valid_raw_config()
    del cfg["mesh"]
    cfg_path = write_yaml(tmp_path / "invalid.yaml", cfg)

    with pytest.raises(ConfigValidationError, match="invalid.yaml"):
        load_and_validate_config(cfg_path)


def test_validate_loaded_config_wraps_schema_base_error_with_source_path(monkeypatch: pytest.MonkeyPatch) -> None:
    def _raise_schema_base_error(raw_cfg: dict) -> None:
        raise ConfigSchemaError("synthetic schema base failure")

    monkeypatch.setattr(config_loader, "validate_config_schema", _raise_schema_base_error)

    with pytest.raises(ConfigValidationError, match="synthetic schema base failure"):
        validate_loaded_config({"case": {"name": "x"}}, source_path=Path("bad.yaml"))


def test_yaml_parse_error_includes_source_path(tmp_path: Path) -> None:
    cfg_path = write_text(tmp_path / "broken.yaml", "case:\n  name: [oops\n")

    with pytest.raises(ConfigLoadError, match="broken.yaml"):
        load_raw_config(cfg_path)
