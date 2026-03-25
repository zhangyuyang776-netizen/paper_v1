from __future__ import annotations

import pytest

from core.config_schema import ConfigValidationError, validate_config_schema


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


def test_missing_required_section_raises() -> None:
    cfg = make_minimal_valid_raw_config()
    del cfg["mesh"]

    with pytest.raises(ConfigValidationError, match="Missing required section: 'mesh'"):
        validate_config_schema(cfg)


def test_unknown_top_level_section_raises() -> None:
    cfg = make_minimal_valid_raw_config()
    cfg["physics"] = {}

    with pytest.raises(ConfigValidationError, match="Unknown top-level section: 'physics'"):
        validate_config_schema(cfg)


def test_unknown_field_in_section_raises() -> None:
    cfg = make_minimal_valid_raw_config()
    cfg["mesh"]["n_gas_far"] = 100

    with pytest.raises(ConfigValidationError, match="Unknown field 'n_gas_far' in section 'mesh'"):
        validate_config_schema(cfg)


def test_missing_required_field_raises() -> None:
    cfg = make_minimal_valid_raw_config()
    del cfg["paths"]["gas_mechanism_path"]

    with pytest.raises(
        ConfigValidationError,
        match="Missing required field 'gas_mechanism_path' in section 'paths'",
    ):
        validate_config_schema(cfg)


def test_bool_not_accepted_as_int() -> None:
    cfg = make_minimal_valid_raw_config()
    cfg["mesh"]["n_liq"] = True

    with pytest.raises(ConfigValidationError, match="field 'n_liq'"):
        validate_config_schema(cfg)


def test_string_not_accepted_for_float() -> None:
    cfg = make_minimal_valid_raw_config()
    cfg["time_stepper"]["dt_start"] = "1e-6"

    with pytest.raises(ConfigValidationError, match="field 'dt_start'"):
        validate_config_schema(cfg)


def test_int_not_accepted_as_bool() -> None:
    cfg = make_minimal_valid_raw_config()
    cfg["solver_inner_petsc"]["use_fieldsplit"] = 1

    with pytest.raises(ConfigValidationError, match="field 'use_fieldsplit'"):
        validate_config_schema(cfg)


def test_negative_pressure_raises() -> None:
    cfg = make_minimal_valid_raw_config()
    cfg["initialization"]["gas_pressure"] = -1.0

    with pytest.raises(ConfigValidationError, match="field 'gas_pressure'"):
        validate_config_schema(cfg)


def test_a0_must_be_strictly_positive() -> None:
    cfg = make_minimal_valid_raw_config()
    cfg["mesh"]["a0"] = 0.0

    with pytest.raises(ConfigValidationError, match="a0 must be greater than 0"):
        validate_config_schema(cfg)


def test_gas_pressure_must_be_strictly_positive() -> None:
    cfg = make_minimal_valid_raw_config()
    cfg["initialization"]["gas_pressure"] = 0.0

    with pytest.raises(ConfigValidationError, match="gas_pressure must be greater than 0"):
        validate_config_schema(cfg)


def test_far_stretch_ratio_less_than_one_raises() -> None:
    cfg = make_minimal_valid_raw_config()
    cfg["mesh"]["far_stretch_ratio"] = 0.9

    with pytest.raises(ConfigValidationError, match="field 'far_stretch_ratio'"):
        validate_config_schema(cfg)


def test_mesh_cross_field_rule_raises() -> None:
    cfg = make_minimal_valid_raw_config()
    cfg["mesh"]["r_end"] = cfg["mesh"]["a0"]

    with pytest.raises(ConfigValidationError, match="r_end must be greater than a0"):
        validate_config_schema(cfg)


def test_dt_order_cross_field_rule_raises() -> None:
    cfg = make_minimal_valid_raw_config()
    cfg["time_stepper"]["dt_min"] = cfg["time_stepper"]["dt_start"]

    with pytest.raises(ConfigValidationError, match="dt_min < dt_start <= dt_max"):
        validate_config_schema(cfg)


def test_shrink_factor_must_be_positive() -> None:
    cfg = make_minimal_valid_raw_config()
    cfg["time_stepper"]["shrink_factor"] = 0.0

    with pytest.raises(ConfigValidationError, match="shrink_factor must satisfy 0 < value <= 1"):
        validate_config_schema(cfg)


def test_corrector_relaxation_must_be_positive_if_given() -> None:
    cfg = make_minimal_valid_raw_config()
    cfg["outer_stepper"]["corrector_relaxation"] = 0.0

    with pytest.raises(ConfigValidationError, match="corrector_relaxation"):
        validate_config_schema(cfg)


def test_recovery_temperature_bounds_raises() -> None:
    cfg = make_minimal_valid_raw_config()
    cfg["recovery"]["T_min_l"] = 900.0
    cfg["recovery"]["T_max_l"] = 800.0

    with pytest.raises(ConfigValidationError, match="T_max_l must be greater than T_min_l"):
        validate_config_schema(cfg)


def test_gas_composition_requires_string_number_mapping() -> None:
    cfg = make_minimal_valid_raw_config()
    cfg["initialization"]["gas_composition"] = {"O2": "0.21", "N2": 0.79}

    with pytest.raises(ConfigValidationError, match="gas_composition"):
        validate_config_schema(cfg)


def test_gas_composition_sum_must_be_one() -> None:
    cfg = make_minimal_valid_raw_config()
    cfg["initialization"]["gas_composition"] = {"O2": 0.20, "N2": 0.70}

    with pytest.raises(ConfigValidationError, match="gas_composition"):
        validate_config_schema(cfg)


def test_liquid_composition_sum_must_be_one() -> None:
    cfg = make_minimal_valid_raw_config()
    cfg["initialization"]["liquid_composition"] = {"ethanol": 0.8}

    with pytest.raises(ConfigValidationError, match="liquid_composition"):
        validate_config_schema(cfg)


def test_y_vap_if0_sum_cannot_exceed_one() -> None:
    cfg = make_minimal_valid_raw_config()
    cfg["initialization"]["Y_vap_if0"] = {"C2H5OH": 0.8, "H2O": 0.4}

    with pytest.raises(ConfigValidationError, match="Y_vap_if0"):
        validate_config_schema(cfg)


def test_liquid_to_gas_species_map_requires_string_string_mapping() -> None:
    cfg = make_minimal_valid_raw_config()
    cfg["species"]["liquid_to_gas_species_map"] = {"ethanol": 123}

    with pytest.raises(ConfigValidationError, match="liquid_to_gas_species_map"):
        validate_config_schema(cfg)


def test_liquid_to_gas_species_map_must_cover_all_liquid_species() -> None:
    cfg = make_minimal_valid_raw_config()
    cfg["initialization"]["liquid_composition"] = {"A": 0.5, "B": 0.5}
    cfg["species"]["liquid_closure_species"] = "B"
    cfg["species"]["liquid_to_gas_species_map"] = {"A": "A_g"}

    with pytest.raises(
        ConfigValidationError,
        match="liquid_to_gas_species_map keys must match initialization.liquid_composition species exactly",
    ):
        validate_config_schema(cfg)


def test_multicomponent_liquid_requires_closure_species() -> None:
    cfg = make_minimal_valid_raw_config()
    cfg["initialization"]["liquid_composition"] = {"A": 0.5, "B": 0.5}
    cfg["species"]["liquid_closure_species"] = None
    cfg["species"]["liquid_to_gas_species_map"] = {"A": "A_g", "B": "B_g"}

    with pytest.raises(ConfigValidationError, match="liquid_closure_species"):
        validate_config_schema(cfg)


def test_multicomponent_liquid_closure_must_belong_to_liquid_species() -> None:
    cfg = make_minimal_valid_raw_config()
    cfg["initialization"]["liquid_composition"] = {"A": 0.5, "B": 0.5}
    cfg["species"]["liquid_closure_species"] = "C"
    cfg["species"]["liquid_to_gas_species_map"] = {"A": "A_g", "B": "B_g"}

    with pytest.raises(
        ConfigValidationError,
        match="liquid_closure_species must be one of initialization.liquid_composition species",
    ):
        validate_config_schema(cfg)


def test_single_component_liquid_allows_none_closure() -> None:
    cfg = make_minimal_valid_raw_config()
    cfg["initialization"]["liquid_composition"] = {"ethanol": 1.0}
    cfg["species"]["liquid_closure_species"] = None

    validate_config_schema(cfg)


def test_snapshot_format_rejects_unsupported_values() -> None:
    cfg = make_minimal_valid_raw_config()
    cfg["output"]["snapshot_format"] = "h5"

    with pytest.raises(ConfigValidationError, match="field 'snapshot_format'"):
        validate_config_schema(cfg)


def test_minimal_valid_config_passes() -> None:
    cfg = make_minimal_valid_raw_config()
    validate_config_schema(cfg)


def test_multicomponent_valid_config_passes() -> None:
    cfg = make_minimal_valid_raw_config()
    cfg["initialization"]["liquid_composition"] = {
        "A": 0.7,
        "B": 0.3,
    }
    cfg["species"]["liquid_closure_species"] = "B"
    cfg["species"]["liquid_to_gas_species_map"] = {
        "A": "A_g",
        "B": "B_g",
    }

    validate_config_schema(cfg)
