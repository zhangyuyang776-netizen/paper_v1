from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import math


class ConfigSchemaError(ValueError):
    """Base error for raw config schema validation."""


class ConfigValidationError(ConfigSchemaError):
    """Raised when raw config violates the paper_v1 schema."""


@dataclass(slots=True, frozen=True)
class SchemaField:
    """Schema descriptor for a single raw-config field."""

    name: str
    expected_type: type | tuple[type, ...]
    required: bool = True
    allow_none: bool = False
    choices: tuple[Any, ...] | None = None
    min_value: float | int | None = None
    max_value: float | int | None = None
    nonempty: bool = False


@dataclass(slots=True, frozen=True)
class SchemaSection:
    """Schema descriptor for one top-level raw-config section."""

    name: str
    required: bool
    fields: tuple[SchemaField, ...]
    allow_unknown_keys: bool = False


def _expect_mapping(name: str, obj: Any) -> Mapping[str, Any]:
    if not isinstance(obj, Mapping):
        raise ConfigValidationError(f"Section '{name}' must be a mapping, got {type(obj).__name__}")
    return obj


def _is_valid_scalar_type(value: Any, expected_type: type) -> bool:
    if expected_type is bool:
        return isinstance(value, bool)
    if expected_type is int:
        return isinstance(value, int) and not isinstance(value, bool)
    if expected_type is float:
        return (isinstance(value, (int, float)) and not isinstance(value, bool))
    return isinstance(value, expected_type)


def _validate_field_type(section: str, field: SchemaField, value: Any) -> None:
    expected = field.expected_type
    if isinstance(expected, tuple):
        if not any(_is_valid_scalar_type(value, item) for item in expected):
            raise ConfigValidationError(
                f"Invalid type for field '{field.name}' in section '{section}': "
                f"got {type(value).__name__}"
            )
        return

    if not _is_valid_scalar_type(value, expected):
        raise ConfigValidationError(
            f"Invalid type for field '{field.name}' in section '{section}': "
            f"got {type(value).__name__}"
        )


def _validate_field_choices(section: str, field: SchemaField, value: Any) -> None:
    if field.choices is not None and value not in field.choices:
        raise ConfigValidationError(
            f"Invalid value for field '{field.name}' in section '{section}': "
            f"expected one of {field.choices}"
        )


def _validate_field_bounds(section: str, field: SchemaField, value: Any) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return
    if not math.isfinite(float(value)):
        raise ConfigValidationError(
            f"Invalid value for field '{field.name}' in section '{section}': must be finite"
        )
    if field.min_value is not None and value < field.min_value:
        raise ConfigValidationError(
            f"Invalid value for field '{field.name}' in section '{section}': "
            f"must be >= {field.min_value}"
        )
    if field.max_value is not None and value > field.max_value:
        raise ConfigValidationError(
            f"Invalid value for field '{field.name}' in section '{section}': "
            f"must be <= {field.max_value}"
        )


def _validate_nonempty(section: str, field: SchemaField, value: Any) -> None:
    if not field.nonempty:
        return
    if isinstance(value, str) and value.strip() == "":
        raise ConfigValidationError(
            f"Field '{field.name}' in section '{section}' must be non-empty"
        )
    if isinstance(value, (list, tuple, dict)) and len(value) == 0:
        raise ConfigValidationError(
            f"Field '{field.name}' in section '{section}' must be non-empty"
        )


def _build_case_schema() -> SchemaSection:
    return SchemaSection(
        name="case",
        required=True,
        fields=(
            SchemaField("name", str, required=True, nonempty=True),
            SchemaField("description", str, required=False, nonempty=True),
        ),
    )


def _build_paths_schema() -> SchemaSection:
    return SchemaSection(
        name="paths",
        required=True,
        fields=(
            SchemaField("liquid_database_path", str, required=True, nonempty=True),
            SchemaField("gas_mechanism_path", str, required=True, nonempty=True),
            SchemaField("output_root", str, required=True, nonempty=True),
        ),
    )


def _build_mesh_schema() -> SchemaSection:
    return SchemaSection(
        name="mesh",
        required=True,
        fields=(
            SchemaField("a0", float, required=True, min_value=0.0),
            SchemaField("r_end", float, required=True, min_value=0.0),
            SchemaField("n_liq", int, required=True, min_value=1),
            SchemaField("n_gas_near", int, required=True, min_value=1),
            SchemaField("far_stretch_ratio", float, required=True, min_value=1.0),
        ),
    )


def _build_species_schema() -> SchemaSection:
    return SchemaSection(
        name="species",
        required=True,
        fields=(
            SchemaField("gas_closure_species", str, required=True, nonempty=True),
            SchemaField("liquid_closure_species", str, required=False, allow_none=True, nonempty=True),
            SchemaField("liquid_to_gas_species_map", dict, required=True, nonempty=True),
        ),
    )


def _build_initialization_schema() -> SchemaSection:
    return SchemaSection(
        name="initialization",
        required=True,
        fields=(
            SchemaField("gas_temperature", float, required=True, min_value=0.0),
            SchemaField("gas_pressure", float, required=True, min_value=0.0),
            SchemaField("liquid_temperature", float, required=True, min_value=0.0),
            SchemaField("gas_composition", dict, required=True, nonempty=True),
            SchemaField("liquid_composition", dict, required=True, nonempty=True),
            SchemaField("t_init_T", float, required=True, min_value=0.0),
            SchemaField("Y_vap_if0", dict, required=True, nonempty=True),
        ),
    )


def _build_time_stepper_schema() -> SchemaSection:
    return SchemaSection(
        name="time_stepper",
        required=True,
        fields=(
            SchemaField("t0", float, required=True),
            SchemaField("t_end", float, required=True),
            SchemaField("dt_start", float, required=True, min_value=0.0),
            SchemaField("dt_min", float, required=True, min_value=0.0),
            SchemaField("dt_max", float, required=True, min_value=0.0),
            SchemaField("retry_max_per_step", int, required=True, min_value=0),
            SchemaField("q_success_for_growth", int, required=True, min_value=1),
            SchemaField("growth_factor", float, required=True, min_value=1.0),
            SchemaField("shrink_factor", float, required=True, min_value=0.0, max_value=1.0),
        ),
    )


def _build_outer_stepper_schema() -> SchemaSection:
    return SchemaSection(
        name="outer_stepper",
        required=True,
        fields=(
            SchemaField("outer_max_iter", int, required=True, min_value=1),
            SchemaField("eps_dot_a_tol", float, required=True, min_value=0.0),
            SchemaField(
                "corrector_relaxation",
                float,
                required=False,
                allow_none=True,
                min_value=0.0,
                max_value=1.0,
            ),
        ),
    )


def _build_solver_inner_petsc_schema() -> SchemaSection:
    return SchemaSection(
        name="solver_inner_petsc",
        required=True,
        fields=(
            SchemaField("inner_max_iter", int, required=True, min_value=1),
            SchemaField("snes_rtol", float, required=True, min_value=0.0),
            SchemaField("snes_atol", float, required=True, min_value=0.0),
            SchemaField("snes_stol", float, required=True, min_value=0.0),
            SchemaField("ksp_rtol", float, required=True, min_value=0.0),
            SchemaField("pc_type", str, required=True, nonempty=True),
            SchemaField("ksp_type", str, required=True, nonempty=True),
            SchemaField("use_fieldsplit", bool, required=True),
        ),
    )


def _build_recovery_schema() -> SchemaSection:
    return SchemaSection(
        name="recovery",
        required=True,
        fields=(
            SchemaField("T_min_l", float, required=True),
            SchemaField("T_max_l", float, required=True),
            SchemaField("T_min_g", float, required=True),
            SchemaField("T_max_g", float, required=True),
            SchemaField("liq_h_inv_tol", float, required=True, min_value=0.0),
            SchemaField("liq_h_inv_max_iter", int, required=True, min_value=1),
            SchemaField("gas_h_inv_tol", float, required=True, min_value=0.0),
            SchemaField("gas_h_inv_max_iter", int, required=True, min_value=1),
            SchemaField("use_cantera_hpy_first", bool, required=True),
        ),
    )


def _build_diagnostics_schema() -> SchemaSection:
    return SchemaSection(
        name="diagnostics",
        required=True,
        fields=(
            SchemaField("verbose_interface_panel", bool, required=True),
            SchemaField("verbose_property_warnings", bool, required=True),
            SchemaField("write_step_diag", bool, required=True),
            SchemaField("write_interface_diag", bool, required=True),
            SchemaField("write_failure_report", bool, required=True),
            SchemaField("output_every_n_steps", int, required=True, min_value=1),
        ),
    )


def _build_output_schema() -> SchemaSection:
    return SchemaSection(
        name="output",
        required=True,
        fields=(
            SchemaField("write_spatial_fields", bool, required=True),
            SchemaField("write_spatial_species", bool, required=True),
            SchemaField("write_time_series_scalars", bool, required=True),
            SchemaField("write_time_series_species", bool, required=True),
            SchemaField("snapshot_format", str, required=True, nonempty=True, choices=("npz",)),
        ),
    )


def _build_validation_schema() -> SchemaSection:
    return SchemaSection(
        name="validation",
        required=True,
        fields=(
            SchemaField("enable_mass_balance_check", bool, required=True),
            SchemaField("enable_energy_balance_check", bool, required=True),
            SchemaField("enable_state_bounds_check", bool, required=True),
        ),
    )


def build_schema() -> dict[str, SchemaSection]:
    return {
        "case": _build_case_schema(),
        "paths": _build_paths_schema(),
        "mesh": _build_mesh_schema(),
        "species": _build_species_schema(),
        "initialization": _build_initialization_schema(),
        "time_stepper": _build_time_stepper_schema(),
        "outer_stepper": _build_outer_stepper_schema(),
        "solver_inner_petsc": _build_solver_inner_petsc_schema(),
        "recovery": _build_recovery_schema(),
        "diagnostics": _build_diagnostics_schema(),
        "output": _build_output_schema(),
        "validation": _build_validation_schema(),
    }


def validate_required_sections(raw_cfg: Mapping[str, Any], schema: Mapping[str, SchemaSection]) -> None:
    for section_name, section_schema in schema.items():
        if section_schema.required and section_name not in raw_cfg:
            raise ConfigValidationError(f"Missing required section: '{section_name}'")


def validate_unknown_top_level_keys(raw_cfg: Mapping[str, Any], schema: Mapping[str, SchemaSection]) -> None:
    for key in raw_cfg:
        if key not in schema:
            raise ConfigValidationError(f"Unknown top-level section: '{key}'")


def validate_section_fields(
    section_name: str,
    section_data: Mapping[str, Any],
    section_schema: SchemaSection,
) -> None:
    data = _expect_mapping(section_name, section_data)
    field_map = {field.name: field for field in section_schema.fields}

    for field in section_schema.fields:
        if field.required and field.name not in data:
            raise ConfigValidationError(
                f"Missing required field '{field.name}' in section '{section_name}'"
            )

    if not section_schema.allow_unknown_keys:
        for key in data:
            if key not in field_map:
                raise ConfigValidationError(
                    f"Unknown field '{key}' in section '{section_name}'"
                )

    for field_name, value in data.items():
        field = field_map[field_name]
        if value is None:
            if not field.allow_none:
                raise ConfigValidationError(
                    f"Field '{field.name}' in section '{section_name}' must not be None"
                )
            continue
        _validate_field_type(section_name, field, value)
        _validate_field_choices(section_name, field, value)
        _validate_field_bounds(section_name, field, value)
        _validate_nonempty(section_name, field, value)


def _validate_string_number_mapping(
    section: str,
    field_name: str,
    mapping: Mapping[str, Any],
    *,
    allow_zero: bool = True,
) -> None:
    """Validate mass-fraction style string->number mappings used in raw initialization input."""
    for key, value in mapping.items():
        if not isinstance(key, str) or key == "":
            raise ConfigValidationError(
                f"Field '{field_name}' in section '{section}' must use non-empty string keys"
            )
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigValidationError(
                f"Field '{field_name}' in section '{section}' must use numeric values"
            )
        if not math.isfinite(float(value)):
            raise ConfigValidationError(
                f"Field '{field_name}' in section '{section}' must use finite numeric values"
            )
        if value < 0.0 or (not allow_zero and value == 0.0):
            cmp = "positive" if not allow_zero else "non-negative"
            raise ConfigValidationError(
                f"Field '{field_name}' in section '{section}' must use {cmp} numeric values"
            )


def _validate_string_string_mapping(
    section: str,
    field_name: str,
    mapping: Mapping[str, Any],
) -> None:
    for key, value in mapping.items():
        if not isinstance(key, str) or key == "":
            raise ConfigValidationError(
                f"Field '{field_name}' in section '{section}' must use non-empty string keys"
            )
        if not isinstance(value, str) or value == "":
            raise ConfigValidationError(
                f"Field '{field_name}' in section '{section}' must use non-empty string values"
            )


def _validate_fraction_mapping_sum(
    section: str,
    field_name: str,
    mapping: Mapping[str, Any],
    *,
    tol: float = 1.0e-10,
) -> None:
    total = float(sum(float(value) for value in mapping.values()))
    if abs(total - 1.0) > tol:
        raise ConfigValidationError(
            f"Field '{field_name}' in section '{section}' must sum to 1 within tolerance {tol}"
        )


def _validate_fraction_mapping_sum_leq_one(
    section: str,
    field_name: str,
    mapping: Mapping[str, Any],
    *,
    tol: float = 1.0e-10,
) -> None:
    total = float(sum(float(value) for value in mapping.values()))
    if total < -tol or total > 1.0 + tol:
        raise ConfigValidationError(
            f"Field '{field_name}' in section '{section}' must satisfy 0 <= sum <= 1 within tolerance {tol}"
        )


def validate_cross_field_rules(raw_cfg: Mapping[str, Any]) -> None:
    mesh = _expect_mapping("mesh", raw_cfg["mesh"])
    if float(mesh["a0"]) <= 0.0:
        raise ConfigValidationError("a0 must be greater than 0")
    if float(mesh["r_end"]) <= float(mesh["a0"]):
        raise ConfigValidationError("r_end must be greater than a0")

    time_stepper = _expect_mapping("time_stepper", raw_cfg["time_stepper"])
    if float(time_stepper["t_end"]) <= float(time_stepper["t0"]):
        raise ConfigValidationError("t_end must be greater than t0")
    if not (float(time_stepper["dt_min"]) < float(time_stepper["dt_start"]) <= float(time_stepper["dt_max"])):
        raise ConfigValidationError("dt_min < dt_start <= dt_max must hold")
    if not (0.0 < float(time_stepper["shrink_factor"]) <= 1.0):
        raise ConfigValidationError("shrink_factor must satisfy 0 < value <= 1")

    outer = _expect_mapping("outer_stepper", raw_cfg["outer_stepper"])
    corrector_relaxation = outer.get("corrector_relaxation")
    if corrector_relaxation is not None and not (0.0 < float(corrector_relaxation) <= 1.0):
        raise ConfigValidationError("corrector_relaxation must satisfy 0 < value <= 1")

    recovery = _expect_mapping("recovery", raw_cfg["recovery"])
    if float(recovery["T_max_l"]) <= float(recovery["T_min_l"]):
        raise ConfigValidationError("T_max_l must be greater than T_min_l")
    if float(recovery["T_max_g"]) <= float(recovery["T_min_g"]):
        raise ConfigValidationError("T_max_g must be greater than T_min_g")

    species = _expect_mapping("species", raw_cfg["species"])
    if not isinstance(species["gas_closure_species"], str) or species["gas_closure_species"].strip() == "":
        raise ConfigValidationError("gas_closure_species must be a non-empty string")
    liquid_to_gas_map = _expect_mapping("species.liquid_to_gas_species_map", species["liquid_to_gas_species_map"])
    _validate_string_string_mapping(
        "species",
        "liquid_to_gas_species_map",
        liquid_to_gas_map,
    )
    # Gas closure membership depends on mechanism loading and is checked later in preprocess.

    init = _expect_mapping("initialization", raw_cfg["initialization"])
    if float(init["gas_pressure"]) <= 0.0:
        raise ConfigValidationError("gas_pressure must be greater than 0")
    if float(init["gas_temperature"]) <= 0.0:
        raise ConfigValidationError("gas_temperature must be greater than 0")
    if float(init["liquid_temperature"]) <= 0.0:
        raise ConfigValidationError("liquid_temperature must be greater than 0")
    if float(init["t_init_T"]) <= 0.0:
        raise ConfigValidationError("t_init_T must be greater than 0")
    gas_comp = _expect_mapping("initialization.gas_composition", init["gas_composition"])
    liq_comp = _expect_mapping("initialization.liquid_composition", init["liquid_composition"])
    y_vap_if0 = _expect_mapping("initialization.Y_vap_if0", init["Y_vap_if0"])

    _validate_string_number_mapping("initialization", "gas_composition", gas_comp)
    _validate_fraction_mapping_sum("initialization", "gas_composition", gas_comp)

    _validate_string_number_mapping("initialization", "liquid_composition", liq_comp)
    _validate_fraction_mapping_sum("initialization", "liquid_composition", liq_comp)

    _validate_string_number_mapping("initialization", "Y_vap_if0", y_vap_if0)
    _validate_fraction_mapping_sum_leq_one("initialization", "Y_vap_if0", y_vap_if0)

    liquid_closure_species = species.get("liquid_closure_species")
    liq_species = set(liq_comp.keys())
    map_keys = set(liquid_to_gas_map.keys())
    if map_keys != liq_species:
        raise ConfigValidationError(
            "liquid_to_gas_species_map keys must match initialization.liquid_composition species exactly"
        )
    if len(liq_comp) > 1:
        if not isinstance(liquid_closure_species, str) or liquid_closure_species == "":
            raise ConfigValidationError(
                "liquid_closure_species must be provided for multicomponent liquid"
            )
        if liquid_closure_species not in liq_species:
            raise ConfigValidationError(
                "liquid_closure_species must be one of initialization.liquid_composition species"
            )
    elif liquid_closure_species is not None and liquid_closure_species not in liq_species:
        raise ConfigValidationError(
            "liquid_closure_species must match the single initialization.liquid_composition species"
        )


def validate_config_schema(raw_cfg: Mapping[str, Any]) -> None:
    schema = build_schema()
    top = _expect_mapping("raw_cfg", raw_cfg)
    validate_required_sections(top, schema)
    validate_unknown_top_level_keys(top, schema)

    for section_name, section_schema in schema.items():
        validate_section_fields(section_name, top[section_name], section_schema)

    validate_cross_field_rules(top)


__all__ = [
    "ConfigSchemaError",
    "ConfigValidationError",
    "SchemaField",
    "SchemaSection",
    "build_schema",
    "validate_required_sections",
    "validate_unknown_top_level_keys",
    "validate_section_fields",
    "validate_cross_field_rules",
    "validate_config_schema",
]
