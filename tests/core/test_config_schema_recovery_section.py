from __future__ import annotations

import copy

import pytest

from core.config_schema import ConfigValidationError, validate_config_schema


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def make_minimal_valid_raw_config() -> dict:
    """Return a raw config dict with the smallest valid recovery block."""
    return {
        "case": {"name": "schema_test"},
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
                "iface": {
                    "ksp_type": "preonly",
                    "pc_type": "lu",
                },
            },
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


# ---------------------------------------------------------------------------
# Case 1: full new recovery block passes validation
# ---------------------------------------------------------------------------

def test_new_recovery_schema_passes() -> None:
    cfg = make_minimal_valid_raw_config()
    # Should not raise.
    validate_config_schema(cfg)


# ---------------------------------------------------------------------------
# Case 2: legacy field names are rejected by the schema
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "legacy_key",
    ["liq_h_inv_tol", "liq_h_inv_max_iter", "gas_h_inv_tol"],
)
def test_legacy_recovery_keys_rejected(legacy_key: str) -> None:
    cfg = make_minimal_valid_raw_config()
    # Remove one of the new required fields to make room, then inject legacy key.
    # The schema uses allow_unknown_keys=False so any unexpected key is rejected.
    del cfg["recovery"]["h_abs_tol"]
    cfg["recovery"][legacy_key] = 1.0e-10
    with pytest.raises(ConfigValidationError):
        validate_config_schema(cfg)


# ---------------------------------------------------------------------------
# Case 3: use_cantera_hpy_first: false is rejected at raw-config layer
# ---------------------------------------------------------------------------

def test_use_cantera_hpy_first_false_rejected() -> None:
    cfg = make_minimal_valid_raw_config()
    cfg["recovery"]["use_cantera_hpy_first"] = False
    with pytest.raises(ConfigValidationError, match="use_cantera_hpy_first"):
        validate_config_schema(cfg)


# ---------------------------------------------------------------------------
# Case 4: zero-value thresholds are rejected at raw-config layer (CS-3 check)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "field",
    ["h_abs_tol", "cp_min", "rho_min", "Y_sum_tol", "species_recovery_eps_abs"],
)
def test_zero_threshold_rejected(field: str) -> None:
    cfg = make_minimal_valid_raw_config()
    cfg["recovery"][field] = 0.0
    with pytest.raises(ConfigValidationError):
        validate_config_schema(cfg)


# ---------------------------------------------------------------------------
# Case 5: temperature bound inversion is rejected
# ---------------------------------------------------------------------------

def test_T_max_l_leq_T_min_l_rejected() -> None:
    cfg = make_minimal_valid_raw_config()
    cfg["recovery"]["T_min_l"] = 900.0
    cfg["recovery"]["T_max_l"] = 800.0
    with pytest.raises(ConfigValidationError, match="T_max_l"):
        validate_config_schema(cfg)


def test_T_max_g_leq_T_min_g_rejected() -> None:
    cfg = make_minimal_valid_raw_config()
    cfg["recovery"]["T_min_g"] = 3000.0
    cfg["recovery"]["T_max_g"] = 2000.0
    with pytest.raises(ConfigValidationError, match="T_max_g"):
        validate_config_schema(cfg)
