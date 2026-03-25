from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from core.preprocess import (
    PreprocessError,
    _build_recovery_config,
    _validate_mass_fraction_vector,
    normalize_config,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def make_valid_recovery_raw() -> dict:
    return {
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
    }


def make_minimal_valid_raw_config() -> dict:
    return {
        "case": {"name": "preprocess_recovery_test"},
        "paths": {
            "liquid_database_path": "data/liquid_db.yaml",
            "gas_mechanism_path": "mech/gas.yaml",
            "output_root": "out/",
        },
        "mesh": {
            "a0": 1.0e-4,
            "r_end": 1.0e-2,
            "n_liq": 10,
            "n_gas_near": 20,
            "far_stretch_ratio": 1.03,
        },
        "species": {
            "gas_closure_species": "N2",
            "liquid_closure_species": None,
            "liquid_to_gas_species_map": {"ethanol": "C2H5OH"},
        },
        "initialization": {
            "gas_temperature": 1000.0,
            "gas_pressure": 101325.0,
            "liquid_temperature": 300.0,
            "gas_composition": {"O2": 0.21, "N2": 0.79},
            "liquid_composition": {"ethanol": 1.0},
            "t_init_T": 1.0e-5,
            "Y_vap_if0": {"C2H5OH": 0.10},
        },
        "time_stepper": {
            "t0": 0.0,
            "t_end": 1.0e-3,
            "dt_start": 1.0e-6,
            "dt_min": 1.0e-9,
            "dt_max": 1.0e-4,
            "max_retries_per_step": 4,
            "accept_growth_factor": 1.2,
            "reject_shrink_factor": 0.5,
        },
        "outer_stepper": {
            "outer_max_iter": 4,
            "predictor_mode": "explicit_from_previous_dot_a",
            "corrector_mode": "trapezoidal_fixed_point",
            "omega_a": 1.0,
            "omega_v": 1.0,
            "outer_convergence_mode": "eps_dot_a",
            "outer_convergence_tol": 1.0e-6,
            "eps_ref_dot_a": 1.0e-12,
        },
        "solver_inner_petsc": {
            "snes_type": "newtonls",
            "linesearch_type": "bt",
            "snes_rtol": 1.0e-8,
            "snes_atol": 1.0e-10,
            "snes_stol": 1.0e-12,
            "snes_max_it": 50,
            "options_prefix": "",
            "lag_jacobian": -1,
            "lag_preconditioner": -1,
            "ksp_type": "fgmres",
            "pc_type": "ilu",
            "ksp_rtol": 1.0e-8,
            "ksp_atol": 1.0e-12,
            "ksp_max_it": 200,
            "restart": 50,
            "gmres_modified_gram_schmidt": True,
            "gmres_preallocate": True,
            "fieldsplit": {
                "scheme": "bulk_iface",
                "type": "schur",
                "schur_fact_type": "full",
                "schur_precondition": "a11",
                "bulk": {
                    "ksp_type": "fgmres",
                    "pc_type": "asm",
                    "sub_ksp_type": "preonly",
                    "sub_pc_type": "ilu",
                    "asm_overlap": 1,
                },
                "iface": {"ksp_type": "preonly", "pc_type": "lu"},
            },
        },
        "recovery": make_valid_recovery_raw(),
        "diagnostics": {
            "verbose_interface_panel": False,
            "verbose_property_warnings": True,
            "write_step_diag": True,
            "write_interface_diag": True,
            "write_failure_report": True,
            "output_every_n_steps": 1,
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


# ---------------------------------------------------------------------------
# Case 1: _build_recovery_config produces correct RecoveryConfig
# ---------------------------------------------------------------------------

def test_build_recovery_config_fields() -> None:
    raw = make_valid_recovery_raw()
    cfg = _build_recovery_config(raw)
    assert cfg.rho_min == 1.0e-12
    assert cfg.h_check_tol == 1.0e-8
    assert cfg.cp_min == 1.0
    assert cfg.liquid_h_inv_max_iter == 50
    assert cfg.gas_h_inv_max_iter == 50
    assert cfg.use_cantera_hpy_first is True


# ---------------------------------------------------------------------------
# Case 2: normalize_config returns RunConfig with new recovery contract
# ---------------------------------------------------------------------------

def test_normalize_config_recovery_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case_dir, _, _ = _prepare_case_files(tmp_path)
    _patch_gas_species(monkeypatch, ("N2", "O2", "C2H5OH"))
    raw_cfg = make_minimal_valid_raw_config()

    run_cfg = normalize_config(raw_cfg, source_path=case_dir / "case.yaml")

    assert run_cfg.recovery.rho_min == 1.0e-12
    assert run_cfg.recovery.h_check_tol == 1.0e-8
    assert run_cfg.recovery.cp_min == 1.0
    assert run_cfg.recovery.Y_sum_tol == 1.0e-10


# ---------------------------------------------------------------------------
# Case 3: _validate_mass_fraction_vector — sum != 1 raises for eq_one
# ---------------------------------------------------------------------------

def test_validate_mf_vector_eq_one_sum_too_large() -> None:
    vec = np.array([0.5, 0.6])  # sum = 1.1
    with pytest.raises(PreprocessError, match="gas_y_full_0"):
        _validate_mass_fraction_vector(
            name="gas_y_full_0",
            vec=vec,
            sum_mode="eq_one",
            sum_tol=1.0e-10,
            neg_tol=1.0e-14,
        )


def test_validate_mf_vector_eq_one_sum_correct_passes() -> None:
    vec = np.array([0.21, 0.79])
    _validate_mass_fraction_vector(
        name="gas_y_full_0",
        vec=vec,
        sum_mode="eq_one",
        sum_tol=1.0e-10,
        neg_tol=1.0e-14,
    )


def test_validate_mf_vector_liquid_sum_wrong_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case_dir, _, _ = _prepare_case_files(tmp_path)
    _patch_gas_species(monkeypatch, ("N2", "O2", "C2H5OH"))
    raw_cfg = make_minimal_valid_raw_config()
    # Make liquid composition sum != 1
    raw_cfg["initialization"]["liquid_composition"] = {"ethanol": 0.5}

    with pytest.raises(PreprocessError, match="liquid_y_full_0"):
        normalize_config(raw_cfg, source_path=case_dir / "case.yaml")


# ---------------------------------------------------------------------------
# Case 4: _validate_mass_fraction_vector — leq_one mode
# ---------------------------------------------------------------------------

def test_validate_mf_vector_leq_one_over_one_raises() -> None:
    vec = np.array([0.6, 0.6])  # sum = 1.2 > 1
    with pytest.raises(PreprocessError, match="y_vap_if0"):
        _validate_mass_fraction_vector(
            name="y_vap_if0",
            vec=vec,
            sum_mode="leq_one",
            sum_tol=1.0e-10,
            neg_tol=1.0e-14,
        )


def test_validate_mf_vector_leq_one_partial_passes() -> None:
    vec = np.array([0.1, 0.0, 0.0])
    _validate_mass_fraction_vector(
        name="y_vap_if0",
        vec=vec,
        sum_mode="leq_one",
        sum_tol=1.0e-10,
        neg_tol=1.0e-14,
    )


# ---------------------------------------------------------------------------
# Case 5: negative values below neg_tol are rejected even in preprocess
# ---------------------------------------------------------------------------

def test_validate_mf_vector_negative_below_tol_raises() -> None:
    neg_tol = 1.0e-14
    vec = np.array([1.0 + 2.0e-14, -2.0e-14])  # second component < -neg_tol
    with pytest.raises(PreprocessError, match="gas_y_full_0"):
        _validate_mass_fraction_vector(
            name="gas_y_full_0",
            vec=vec,
            sum_mode="eq_one",
            sum_tol=1.0e-10,
            neg_tol=neg_tol,
        )


def test_validate_mf_vector_tiny_negative_within_tol_passes() -> None:
    """Values in [-neg_tol, 0) are allowed (no silent clip, but no error)."""
    neg_tol = 1.0e-14
    vec = np.array([1.0 + 5.0e-15, -5.0e-15])  # second component within tol
    _validate_mass_fraction_vector(
        name="gas_y_full_0",
        vec=vec,
        sum_mode="eq_one",
        sum_tol=1.0e-10,
        neg_tol=neg_tol,
    )


# ---------------------------------------------------------------------------
# Case 6: normal full config round-trip succeeds
# ---------------------------------------------------------------------------

def test_full_normalize_config_round_trip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case_dir, _, _ = _prepare_case_files(tmp_path)
    _patch_gas_species(monkeypatch, ("N2", "O2", "C2H5OH"))
    raw_cfg = make_minimal_valid_raw_config()

    run_cfg = normalize_config(raw_cfg, source_path=case_dir / "case.yaml")

    assert run_cfg.recovery.use_cantera_hpy_first is True
    assert run_cfg.initialization.gas_temperature == 1000.0
    np.testing.assert_allclose(
        run_cfg.initialization.gas_y_full_0.sum(), 1.0, atol=1.0e-10
    )
