from __future__ import annotations

"""Load and validate the liquid pure-species parameter database for Phase 2.

This module intentionally stays narrow: it only manages the liquid database,
normalizes validated records into read-only dataclasses, and serves query
helpers for later property/equilibrium/recovery modules.

Current Phase 2 v1 support is intentionally conservative:
- cp_model: ``shomate``
- rho_model / k_model / mu_model: ``merino_log_poly``
"""

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np
import yaml

from core.types import RunConfig


class LiquidDatabaseError(Exception):
    """Base error for liquid database loading and lookup."""


class LiquidDatabaseFormatError(LiquidDatabaseError):
    """Raised when the raw database file does not match the expected top-level format."""


class LiquidDatabaseValidationError(LiquidDatabaseError):
    """Raised when a liquid database entry is structurally invalid."""


class LiquidSpeciesNotFoundError(LiquidDatabaseError):
    """Raised when a requested liquid species cannot be found in the database."""


class LiquidDatabaseRangeError(LiquidDatabaseError):
    """Raised when requested liquid species do not share a valid common temperature range."""


@dataclass(frozen=True)
class LiquidPureSpeciesRecord:
    name: str
    aliases: tuple[str, ...]
    molecular_weight: float
    boiling_temperature: float
    molar_volume: float
    association_factor: float
    T_ref: float
    cp_model: str
    cp_T_range: tuple[float, float]
    cp_coeffs: Mapping[str, float]
    hvap_ref: float
    Tc: float | None
    hvap_model: str | None
    hvap_coeffs: Mapping[str, float]
    rho_model: str
    rho_coeffs: Mapping[str, float]
    k_model: str
    k_coeffs: Mapping[str, float]
    mu_model: str
    mu_coeffs: Mapping[str, float]
    activity_model: str | None
    unifac_groups: Mapping[str, float] | None
    boiling_temperature_atm: float | None = None
    pressure_banks: tuple["LiquidPressureBank", ...] = ()


@dataclass(frozen=True)
class LiquidPressureBank:
    p_fit: float
    boiling_temperature: float
    T_ref: float
    cp_model: str
    cp_T_range: tuple[float, float]
    cp_coeffs: Mapping[str, float]
    hvap_ref: float
    Tc: float | None
    hvap_model: str | None
    hvap_coeffs: Mapping[str, float]
    rho_model: str
    rho_coeffs: Mapping[str, float]
    k_model: str
    k_coeffs: Mapping[str, float]
    mu_model: str
    mu_coeffs: Mapping[str, float]


@dataclass(frozen=True)
class LiquidDBMeta:
    file_type: str
    version: int
    units: Mapping[str, str]
    reference_T: float


@dataclass(frozen=True)
class LiquidDBGlobalModels:
    liquid_cp_default_model: str
    liquid_density_default_model: str
    liquid_conductivity_default_model: str
    liquid_viscosity_default_model: str
    liquid_diffusion_model: str
    liquid_mixture_density_model: str
    liquid_mixture_conductivity_model: str
    liquid_mixture_viscosity_model: str
    activity_model_default: str | None


@dataclass(frozen=True)
class LiquidDatabase:
    meta: LiquidDBMeta
    global_models: LiquidDBGlobalModels
    unifac: Mapping[str, object] | None
    species_by_name: Mapping[str, LiquidPureSpeciesRecord]
    alias_to_name: Mapping[str, str]


_SHOMATE_KEYS = ("A", "B", "C", "D", "E", "F", "G", "H")
_MERINO_KEYS = ("A", "B", "C", "D", "E", "F")


def _freeze_value(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({str(k): _freeze_value(v) for k, v in value.items()})
    if isinstance(value, list):
        return tuple(_freeze_value(item) for item in value)
    return value


def _expect_mapping(name: str, value: Any, *, path: Path | None = None) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        location = f" in {path}" if path is not None else ""
        raise LiquidDatabaseFormatError(f"{name} must be a mapping{location}")
    return value


def _expect_nonempty_str(name: str, value: Any, *, path: Path | None = None) -> str:
    if not isinstance(value, str) or value.strip() == "":
        location = f" in {path}" if path is not None else ""
        raise LiquidDatabaseValidationError(f"{name} must be a non-empty string{location}")
    return value


def _expect_positive_float(name: str, value: Any, *, path: Path | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or value <= 0.0:
        location = f" in {path}" if path is not None else ""
        raise LiquidDatabaseValidationError(f"{name} must be a finite positive number{location}")
    return float(value)


def _expect_optional_positive_float(name: str, value: Any, *, path: Path | None = None) -> float | None:
    if value is None:
        return None
    return _expect_positive_float(name, value, path=path)


def _expect_string_mapping(
    name: str,
    value: Any,
    *,
    path: Path | None = None,
) -> Mapping[str, str]:
    mapping = _expect_mapping(name, value, path=path)
    out: dict[str, str] = {}
    for key, item in mapping.items():
        out[str(_expect_nonempty_str(f"{name} key", key, path=path))] = _expect_nonempty_str(
            f"{name}[{key!r}]",
            item,
            path=path,
        )
    return MappingProxyType(out)


def _expect_float_mapping(
    name: str,
    value: Any,
    *,
    required_keys: tuple[str, ...] | None = None,
    path: Path | None = None,
) -> Mapping[str, float]:
    mapping = _expect_mapping(name, value, path=path)
    out: dict[str, float] = {}
    for key, item in mapping.items():
        out[str(_expect_nonempty_str(f"{name} key", key, path=path))] = _expect_positive_or_finite_number(
            f"{name}[{key!r}]",
            item,
            path=path,
        )
    if required_keys is not None:
        missing = [key for key in required_keys if key not in out]
        if missing:
            location = f" in {path}" if path is not None else ""
            raise LiquidDatabaseValidationError(f"{name} is missing required coefficients {missing}{location}")
    return MappingProxyType(out)


def _expect_nonnegative_count_mapping(
    name: str,
    value: Any,
    *,
    path: Path | None = None,
) -> Mapping[str, float]:
    mapping = _expect_mapping(name, value, path=path)
    out: dict[str, float] = {}
    for key, item in mapping.items():
        key_name = str(_expect_nonempty_str(f"{name} key", key, path=path))
        number = _expect_positive_or_finite_number(f"{name}[{key!r}]", item, path=path)
        if number < 0.0:
            location = f" in {path}" if path is not None else ""
            raise LiquidDatabaseValidationError(f"{name}[{key!r}] must be non-negative{location}")
        if not np.isclose(number, round(number), rtol=0.0, atol=1.0e-12):
            location = f" in {path}" if path is not None else ""
            raise LiquidDatabaseValidationError(f"{name}[{key!r}] must be an integer-like group count{location}")
        out[key_name] = float(round(number))
    return MappingProxyType(out)


def _expect_positive_or_finite_number(name: str, value: Any, *, path: Path | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
        location = f" in {path}" if path is not None else ""
        raise LiquidDatabaseValidationError(f"{name} must be a finite numeric value{location}")
    return float(value)


def _parse_cp_range(value: Any, *, path: Path | None = None) -> tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        location = f" in {path}" if path is not None else ""
        raise LiquidDatabaseValidationError(f"cp_T_range must be a length-2 sequence{location}")
    t_min = _expect_positive_float("cp_T_range[0]", value[0], path=path)
    t_max = _expect_positive_float("cp_T_range[1]", value[1], path=path)
    if t_max <= t_min:
        location = f" in {path}" if path is not None else ""
        raise LiquidDatabaseValidationError(f"cp_T_range must satisfy Tmin < Tmax{location}")
    return (t_min, t_max)


def _parse_pressure_bank(
    raw_bank: Any,
    *,
    species_name: str,
    default_Tc: float | None,
    path: Path,
) -> LiquidPressureBank:
    bank = _expect_mapping(f"{species_name}.pressure_bank", raw_bank, path=path)
    p_fit = _expect_positive_float(f"{species_name}.pressure_bank.p_fit", bank.get("p_fit"), path=path)
    boiling_temperature = _expect_positive_float(
        f"{species_name}.pressure_bank.boiling_temperature",
        bank.get("boiling_temperature"),
        path=path,
    )
    T_ref = _expect_positive_float(f"{species_name}.pressure_bank.T_ref", bank.get("T_ref"), path=path)

    cp_model = _expect_nonempty_str(f"{species_name}.pressure_bank.cp_model", bank.get("cp_model"), path=path)
    cp_T_range = _parse_cp_range(bank.get("cp_T_range"), path=path)
    cp_coeffs = _expect_float_mapping(
        f"{species_name}.pressure_bank.cp_coeffs",
        bank.get("cp_coeffs"),
        required_keys=_SHOMATE_KEYS if cp_model == "shomate" else None,
        path=path,
    )
    if cp_model != "shomate":
        raise LiquidDatabaseValidationError(
            f"{species_name}.pressure_bank.cp_model '{cp_model}' is not supported in the current version"
        )

    hvap_ref = _expect_positive_float(f"{species_name}.pressure_bank.hvap_ref", bank.get("hvap_ref"), path=path)
    hvap_model = bank.get("hvap_model")
    if hvap_model is not None:
        hvap_model = _expect_nonempty_str(f"{species_name}.pressure_bank.hvap_model", hvap_model, path=path)
    Tc = _expect_optional_positive_float(f"{species_name}.pressure_bank.Tc", bank.get("Tc"), path=path)
    if Tc is None:
        Tc = default_Tc
    hvap_coeffs = _expect_float_mapping(
        f"{species_name}.pressure_bank.hvap_coeffs",
        bank.get("hvap_coeffs", {}),
        path=path,
    )
    if hvap_model is not None and hvap_model.lower() == "watson":
        if Tc is None:
            raise LiquidDatabaseValidationError(f"{species_name}.Tc must be provided when hvap_model is watson")
        if "exponent" not in hvap_coeffs:
            raise LiquidDatabaseValidationError(f"{species_name}.pressure_bank.hvap_coeffs must contain 'exponent'")

    rho_model = _expect_nonempty_str(f"{species_name}.pressure_bank.rho_model", bank.get("rho_model"), path=path)
    rho_coeffs = _expect_float_mapping(
        f"{species_name}.pressure_bank.rho_coeffs",
        bank.get("rho_coeffs"),
        required_keys=_MERINO_KEYS if rho_model == "merino_log_poly" else None,
        path=path,
    )
    if rho_model != "merino_log_poly":
        raise LiquidDatabaseValidationError(
            f"{species_name}.pressure_bank.rho_model '{rho_model}' is not supported in the current version"
        )

    k_model = _expect_nonempty_str(f"{species_name}.pressure_bank.k_model", bank.get("k_model"), path=path)
    k_coeffs = _expect_float_mapping(
        f"{species_name}.pressure_bank.k_coeffs",
        bank.get("k_coeffs"),
        required_keys=_MERINO_KEYS if k_model == "merino_log_poly" else None,
        path=path,
    )
    if k_model != "merino_log_poly":
        raise LiquidDatabaseValidationError(
            f"{species_name}.pressure_bank.k_model '{k_model}' is not supported in the current version"
        )

    mu_model = _expect_nonempty_str(f"{species_name}.pressure_bank.mu_model", bank.get("mu_model"), path=path)
    mu_coeffs = _expect_float_mapping(
        f"{species_name}.pressure_bank.mu_coeffs",
        bank.get("mu_coeffs"),
        required_keys=_MERINO_KEYS if mu_model == "merino_log_poly" else None,
        path=path,
    )
    if mu_model != "merino_log_poly":
        raise LiquidDatabaseValidationError(
            f"{species_name}.pressure_bank.mu_model '{mu_model}' is not supported in the current version"
        )

    return LiquidPressureBank(
        p_fit=p_fit,
        boiling_temperature=boiling_temperature,
        T_ref=T_ref,
        cp_model=cp_model,
        cp_T_range=cp_T_range,
        cp_coeffs=cp_coeffs,
        hvap_ref=hvap_ref,
        Tc=Tc,
        hvap_model=hvap_model,
        hvap_coeffs=hvap_coeffs,
        rho_model=rho_model,
        rho_coeffs=rho_coeffs,
        k_model=k_model,
        k_coeffs=k_coeffs,
        mu_model=mu_model,
        mu_coeffs=mu_coeffs,
    )


def _normalize_aliases(value: Any, *, path: Path | None = None) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, (list, tuple)):
        location = f" in {path}" if path is not None else ""
        raise LiquidDatabaseValidationError(f"aliases must be a sequence of strings{location}")
    aliases = tuple(_expect_nonempty_str("alias", item, path=path) for item in value)
    if len(set(aliases)) != len(aliases):
        location = f" in {path}" if path is not None else ""
        raise LiquidDatabaseValidationError(f"aliases must not contain duplicates{location}")
    return aliases


def _activity_uses_unifac(activity_model: str | None) -> bool:
    return activity_model is not None and activity_model.upper() == "UNIFAC"


def _parse_unifac(raw_unifac: Any, *, path: Path) -> Mapping[str, object]:
    unifac = _expect_mapping("unifac", raw_unifac, path=path)
    enabled = unifac.get("enabled")
    if not isinstance(enabled, bool):
        raise LiquidDatabaseValidationError(f"unifac.enabled must be a boolean in {path}")

    raw_groups = _expect_mapping("unifac.groups", unifac.get("groups"), path=path)
    if len(raw_groups) == 0:
        raise LiquidDatabaseValidationError(f"unifac.groups must not be empty in {path}")
    groups_out: dict[str, Mapping[str, float]] = {}
    for group_name, raw_group in raw_groups.items():
        group = _expect_mapping(f"unifac.groups[{group_name!r}]", raw_group, path=path)
        groups_out[_expect_nonempty_str("unifac group name", group_name, path=path)] = MappingProxyType(
            {
                "R": _expect_positive_float(f"unifac.groups[{group_name!r}].R", group.get("R"), path=path),
                "Q": _expect_positive_float(f"unifac.groups[{group_name!r}].Q", group.get("Q"), path=path),
            }
        )

    group_names = tuple(groups_out.keys())
    raw_interactions = _expect_mapping("unifac.interactions", unifac.get("interactions"), path=path)
    if len(raw_interactions) == 0:
        raise LiquidDatabaseValidationError(f"unifac.interactions must not be empty in {path}")
    interactions_out: dict[str, Mapping[str, float]] = {}
    for row_name, raw_row in raw_interactions.items():
        row_key = _expect_nonempty_str("unifac interaction row name", row_name, path=path)
        if row_key not in groups_out:
            raise LiquidDatabaseValidationError(f"unifac.interactions row '{row_key}' is not defined in unifac.groups")
        row = _expect_mapping(f"unifac.interactions[{row_key!r}]", raw_row, path=path)
        parsed_row: dict[str, float] = {}
        for col_name in group_names:
            if col_name not in row:
                raise LiquidDatabaseValidationError(
                    f"unifac.interactions[{row_key!r}] is missing column '{col_name}' in {path}"
                )
            parsed_row[col_name] = _expect_positive_or_finite_number(
                f"unifac.interactions[{row_key!r}][{col_name!r}]",
                row[col_name],
                path=path,
            )
        extra_cols = set(row.keys()) - set(group_names)
        if extra_cols:
            raise LiquidDatabaseValidationError(
                f"unifac.interactions[{row_key!r}] contains unknown columns {sorted(extra_cols)} in {path}"
            )
        interactions_out[row_key] = MappingProxyType(parsed_row)

    missing_rows = set(group_names) - set(interactions_out.keys())
    if missing_rows:
        raise LiquidDatabaseValidationError(f"unifac.interactions is missing rows {sorted(missing_rows)} in {path}")

    return MappingProxyType(
        {
            "enabled": enabled,
            "groups": MappingProxyType(groups_out),
            "interactions": MappingProxyType(interactions_out),
        }
    )


def _parse_meta(raw_meta: Any, *, path: Path) -> LiquidDBMeta:
    meta = _expect_mapping("meta", raw_meta, path=path)
    file_type = _expect_nonempty_str("meta.file_type", meta.get("file_type"), path=path)
    if file_type != "liquid_properties_db":
        raise LiquidDatabaseFormatError(f"meta.file_type must be 'liquid_properties_db' in {path}")
    version = meta.get("version")
    if isinstance(version, bool) or not isinstance(version, int) or version < 1:
        raise LiquidDatabaseFormatError(f"meta.version must be a positive integer in {path}")
    units = _expect_string_mapping("meta.units", meta.get("units"), path=path)
    reference = _expect_mapping("meta.reference", meta.get("reference"), path=path)
    reference_T = _expect_positive_float("meta.reference.T_ref", reference.get("T_ref"), path=path)
    return LiquidDBMeta(
        file_type=file_type,
        version=version,
        units=units,
        reference_T=reference_T,
    )


def _parse_global_models(raw_models: Any, *, path: Path) -> LiquidDBGlobalModels:
    models = _expect_mapping("global_models", raw_models, path=path)
    return LiquidDBGlobalModels(
        liquid_cp_default_model=_expect_nonempty_str(
            "global_models.liquid_cp_default_model",
            models.get("liquid_cp_default_model"),
            path=path,
        ),
        liquid_density_default_model=_expect_nonempty_str(
            "global_models.liquid_density_default_model",
            models.get("liquid_density_default_model"),
            path=path,
        ),
        liquid_conductivity_default_model=_expect_nonempty_str(
            "global_models.liquid_conductivity_default_model",
            models.get("liquid_conductivity_default_model"),
            path=path,
        ),
        liquid_viscosity_default_model=_expect_nonempty_str(
            "global_models.liquid_viscosity_default_model",
            models.get("liquid_viscosity_default_model"),
            path=path,
        ),
        liquid_diffusion_model=_expect_nonempty_str(
            "global_models.liquid_diffusion_model",
            models.get("liquid_diffusion_model"),
            path=path,
        ),
        liquid_mixture_density_model=_expect_nonempty_str(
            "global_models.liquid_mixture_density_model",
            models.get("liquid_mixture_density_model"),
            path=path,
        ),
        liquid_mixture_conductivity_model=_expect_nonempty_str(
            "global_models.liquid_mixture_conductivity_model",
            models.get("liquid_mixture_conductivity_model"),
            path=path,
        ),
        liquid_mixture_viscosity_model=_expect_nonempty_str(
            "global_models.liquid_mixture_viscosity_model",
            models.get("liquid_mixture_viscosity_model"),
            path=path,
        ),
        activity_model_default=(
            None
            if models.get("activity_model_default") is None
            else _expect_nonempty_str(
                "global_models.activity_model_default",
                models.get("activity_model_default"),
                path=path,
            )
        ),
    )


def _parse_species_record(
    raw_record: Any,
    *,
    meta: LiquidDBMeta,
    global_models: LiquidDBGlobalModels,
    has_unifac: bool,
    path: Path,
) -> LiquidPureSpeciesRecord:
    rec = _expect_mapping("species record", raw_record, path=path)
    name = _expect_nonempty_str("species.name", rec.get("name"), path=path)
    aliases = _normalize_aliases(rec.get("aliases"), path=path)
    molecular_weight = _expect_positive_float(f"{name}.molecular_weight", rec.get("molecular_weight"), path=path)
    molar_volume = _expect_positive_float(f"{name}.molar_volume", rec.get("molar_volume"), path=path)
    association_factor = _expect_positive_float(
        f"{name}.association_factor",
        rec.get("association_factor"),
        path=path,
    )
    top_level_Tc = _expect_optional_positive_float(f"{name}.Tc", rec.get("Tc"), path=path)

    activity_model = rec.get("activity_model", global_models.activity_model_default)
    if activity_model is not None:
        activity_model = _expect_nonempty_str(f"{name}.activity_model", activity_model, path=path)

    unifac_groups_raw = rec.get("unifac_groups")
    if unifac_groups_raw is None:
        unifac_groups = None
    else:
        unifac_groups = _expect_nonnegative_count_mapping(f"{name}.unifac_groups", unifac_groups_raw, path=path)
    if _activity_uses_unifac(activity_model):
        if not has_unifac:
            raise LiquidDatabaseValidationError(f"{name} requires UNIFAC data but top-level unifac section is missing")
        if not unifac_groups:
            raise LiquidDatabaseValidationError(f"{name}.unifac_groups must be provided when activity_model is UNIFAC")

    boiling_temperature_atm = _expect_optional_positive_float(
        f"{name}.boiling_temperature_atm",
        rec.get("boiling_temperature_atm"),
        path=path,
    )
    raw_pressure_banks = rec.get("pressure_banks")
    if raw_pressure_banks is not None:
        if boiling_temperature_atm is None:
            boiling_temperature_atm = _expect_positive_float(
                f"{name}.boiling_temperature_atm",
                rec.get("boiling_temperature"),
                path=path,
            )
        if not isinstance(raw_pressure_banks, (list, tuple)) or len(raw_pressure_banks) == 0:
            raise LiquidDatabaseValidationError(f"{name}.pressure_banks must be a non-empty sequence")
        pressure_banks = tuple(
            _parse_pressure_bank(raw_bank, species_name=name, default_Tc=top_level_Tc, path=path)
            for raw_bank in raw_pressure_banks
        )
        p_fit_values = tuple(bank.p_fit for bank in pressure_banks)
        if len(set(p_fit_values)) != len(p_fit_values):
            raise LiquidDatabaseValidationError(f"{name}.pressure_banks must use unique p_fit values")
        pressure_banks = tuple(sorted(pressure_banks, key=lambda bank: bank.p_fit))
        base_bank = pressure_banks[0]
    else:
        if boiling_temperature_atm is None:
            boiling_temperature_atm = _expect_positive_float(
                f"{name}.boiling_temperature_atm",
                rec.get("boiling_temperature"),
                path=path,
            )
        T_ref = _expect_positive_float(f"{name}.T_ref", rec.get("T_ref"), path=path)
        if not np.isclose(T_ref, meta.reference_T, rtol=0.0, atol=1.0e-12):
            raise LiquidDatabaseValidationError(
                f"{name}.T_ref must match database reference temperature {meta.reference_T} in {path}"
            )
        cp_model = _expect_nonempty_str(f"{name}.cp_model", rec.get("cp_model"), path=path)
        cp_T_range = _parse_cp_range(rec.get("cp_T_range"), path=path)
        cp_coeffs = _expect_float_mapping(
            f"{name}.cp_coeffs",
            rec.get("cp_coeffs"),
            required_keys=_SHOMATE_KEYS if cp_model == "shomate" else None,
            path=path,
        )
        if cp_model != "shomate":
            raise LiquidDatabaseValidationError(f"{name}.cp_model '{cp_model}' is not supported in the current version")

        hvap_ref = _expect_positive_float(f"{name}.hvap_ref", rec.get("hvap_ref"), path=path)
        hvap_model = rec.get("hvap_model")
        if hvap_model is not None:
            hvap_model = _expect_nonempty_str(f"{name}.hvap_model", hvap_model, path=path)
        hvap_coeffs = _expect_float_mapping(f"{name}.hvap_coeffs", rec.get("hvap_coeffs", {}), path=path)
        if hvap_model is not None and hvap_model.lower() == "watson":
            if top_level_Tc is None:
                raise LiquidDatabaseValidationError(f"{name}.Tc must be provided when hvap_model is watson")
            if "exponent" not in hvap_coeffs:
                raise LiquidDatabaseValidationError(f"{name}.hvap_coeffs must contain 'exponent' for watson model")

        rho_model = _expect_nonempty_str(f"{name}.rho_model", rec.get("rho_model"), path=path)
        rho_coeffs = _expect_float_mapping(
            f"{name}.rho_coeffs",
            rec.get("rho_coeffs"),
            required_keys=_MERINO_KEYS if rho_model == "merino_log_poly" else None,
            path=path,
        )
        if rho_model != "merino_log_poly":
            raise LiquidDatabaseValidationError(f"{name}.rho_model '{rho_model}' is not supported in the current version")

        k_model = _expect_nonempty_str(f"{name}.k_model", rec.get("k_model"), path=path)
        k_coeffs = _expect_float_mapping(
            f"{name}.k_coeffs",
            rec.get("k_coeffs"),
            required_keys=_MERINO_KEYS if k_model == "merino_log_poly" else None,
            path=path,
        )
        if k_model != "merino_log_poly":
            raise LiquidDatabaseValidationError(f"{name}.k_model '{k_model}' is not supported in the current version")

        mu_model = _expect_nonempty_str(f"{name}.mu_model", rec.get("mu_model"), path=path)
        mu_coeffs = _expect_float_mapping(
            f"{name}.mu_coeffs",
            rec.get("mu_coeffs"),
            required_keys=_MERINO_KEYS if mu_model == "merino_log_poly" else None,
            path=path,
        )
        if mu_model != "merino_log_poly":
            raise LiquidDatabaseValidationError(f"{name}.mu_model '{mu_model}' is not supported in the current version")

        base_bank = LiquidPressureBank(
            p_fit=101325.0,
            boiling_temperature=boiling_temperature_atm,
            T_ref=T_ref,
            cp_model=cp_model,
            cp_T_range=cp_T_range,
            cp_coeffs=cp_coeffs,
            hvap_ref=hvap_ref,
            Tc=top_level_Tc,
            hvap_model=hvap_model,
            hvap_coeffs=hvap_coeffs,
            rho_model=rho_model,
            rho_coeffs=rho_coeffs,
            k_model=k_model,
            k_coeffs=k_coeffs,
            mu_model=mu_model,
            mu_coeffs=mu_coeffs,
        )
        pressure_banks = (base_bank,)

    return LiquidPureSpeciesRecord(
        name=name,
        aliases=aliases,
        molecular_weight=molecular_weight,
        boiling_temperature=boiling_temperature_atm,
        boiling_temperature_atm=boiling_temperature_atm,
        molar_volume=molar_volume,
        association_factor=association_factor,
        T_ref=base_bank.T_ref,
        cp_model=base_bank.cp_model,
        cp_T_range=base_bank.cp_T_range,
        cp_coeffs=base_bank.cp_coeffs,
        hvap_ref=base_bank.hvap_ref,
        Tc=(top_level_Tc if top_level_Tc is not None else base_bank.Tc),
        hvap_model=base_bank.hvap_model,
        hvap_coeffs=base_bank.hvap_coeffs,
        rho_model=base_bank.rho_model,
        rho_coeffs=base_bank.rho_coeffs,
        k_model=base_bank.k_model,
        k_coeffs=base_bank.k_coeffs,
        mu_model=base_bank.mu_model,
        mu_coeffs=base_bank.mu_coeffs,
        activity_model=activity_model,
        unifac_groups=unifac_groups,
        pressure_banks=pressure_banks,
    )


def load_liquid_database(db_path: str | Path) -> LiquidDatabase:
    path = Path(db_path).expanduser().resolve()
    if not path.exists():
        raise LiquidDatabaseError(f"liquid database file does not exist: {path}")
    if not path.is_file():
        raise LiquidDatabaseError(f"liquid database path is not a file: {path}")

    try:
        with path.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
    except yaml.YAMLError as exc:
        raise LiquidDatabaseFormatError(f"failed to parse liquid database YAML '{path}': {exc}") from exc
    except OSError as exc:
        raise LiquidDatabaseError(f"failed to read liquid database '{path}': {exc}") from exc

    if not isinstance(raw, Mapping):
        raise LiquidDatabaseFormatError(f"liquid database root must be a mapping in {path}")

    meta = _parse_meta(raw.get("meta"), path=path)
    global_models = _parse_global_models(raw.get("global_models"), path=path)
    unifac_raw = raw.get("unifac")
    if unifac_raw is None:
        unifac = None
    else:
        unifac = _parse_unifac(unifac_raw, path=path)

    raw_species = raw.get("species")
    if not isinstance(raw_species, list):
        raise LiquidDatabaseFormatError(f"species must be a non-empty list in {path}")
    if len(raw_species) == 0:
        raise LiquidDatabaseFormatError(f"species must not be empty in {path}")

    species_records: dict[str, LiquidPureSpeciesRecord] = {}
    alias_to_name: dict[str, str] = {}
    for raw_record in raw_species:
        record = _parse_species_record(
            raw_record,
            meta=meta,
            global_models=global_models,
            has_unifac=unifac is not None,
            path=path,
        )
        if record.name in species_records:
            raise LiquidDatabaseValidationError(f"duplicate liquid species name '{record.name}' in {path}")
        if record.name in alias_to_name:
            raise LiquidDatabaseValidationError(f"species name '{record.name}' conflicts with an alias in {path}")
        species_records[record.name] = record
        for alias in record.aliases:
            if alias in species_records:
                raise LiquidDatabaseValidationError(f"alias '{alias}' conflicts with a species name in {path}")
            owner = alias_to_name.get(alias)
            if owner is not None:
                raise LiquidDatabaseValidationError(
                    f"alias '{alias}' is duplicated between species '{owner}' and '{record.name}' in {path}"
                )
            alias_to_name[alias] = record.name

    if _activity_uses_unifac(global_models.activity_model_default) and unifac is None:
        raise LiquidDatabaseValidationError(
            f"global_models.activity_model_default requires top-level unifac data in {path}"
        )

    return LiquidDatabase(
        meta=meta,
        global_models=global_models,
        unifac=unifac,
        species_by_name=MappingProxyType(species_records),
        alias_to_name=MappingProxyType(alias_to_name),
    )


def resolve_species_name(db: LiquidDatabase, name_or_alias: str) -> str:
    if name_or_alias in db.species_by_name:
        return name_or_alias
    resolved = db.alias_to_name.get(name_or_alias)
    if resolved is None:
        raise LiquidSpeciesNotFoundError(f"liquid species '{name_or_alias}' was not found in the database")
    return resolved


def get_species_record(db: LiquidDatabase, species_name: str) -> LiquidPureSpeciesRecord:
    canonical = resolve_species_name(db, species_name)
    return db.species_by_name[canonical]


def has_species(db: LiquidDatabase, species_name: str) -> bool:
    try:
        resolve_species_name(db, species_name)
    except LiquidSpeciesNotFoundError:
        return False
    return True


def get_common_cp_temperature_range(db: LiquidDatabase, liquid_species_full: tuple[str, ...]) -> tuple[float, float]:
    return get_common_cp_temperature_range_for_pressure(db, liquid_species_full, p_env=None)


def _legacy_pressure_bank(rec: LiquidPureSpeciesRecord) -> LiquidPressureBank:
    """Build a synthetic single pressure bank for legacy single-bank records."""
    return LiquidPressureBank(
        p_fit=101325.0,
        boiling_temperature=rec.boiling_temperature,
        T_ref=rec.T_ref,
        cp_model=rec.cp_model,
        cp_T_range=rec.cp_T_range,
        cp_coeffs=rec.cp_coeffs,
        hvap_ref=rec.hvap_ref,
        Tc=rec.Tc,
        hvap_model=rec.hvap_model,
        hvap_coeffs=rec.hvap_coeffs,
        rho_model=rec.rho_model,
        rho_coeffs=rec.rho_coeffs,
        k_model=rec.k_model,
        k_coeffs=rec.k_coeffs,
        mu_model=rec.mu_model,
        mu_coeffs=rec.mu_coeffs,
    )


def select_pressure_bank(rec: LiquidPureSpeciesRecord, p_env: float) -> LiquidPressureBank:
    p_env_value = _expect_positive_float("p_env", p_env)
    if len(rec.pressure_banks) == 0:
        return _legacy_pressure_bank(rec)
    distances = np.array([abs(np.log(p_env_value / bank.p_fit)) for bank in rec.pressure_banks], dtype=np.float64)
    return rec.pressure_banks[int(np.argmin(distances))]


def select_pressure_bank_with_diagnostics(
    rec: LiquidPureSpeciesRecord,
    p_env: float,
) -> tuple[LiquidPressureBank, dict[str, float]]:
    bank = select_pressure_bank(rec, p_env)
    p_env_value = _expect_positive_float("p_env", p_env)
    return bank, {
        "selected_p_fit": float(bank.p_fit),
        "selection_distance_log": float(abs(np.log(p_env_value / bank.p_fit))),
    }


def get_common_cp_temperature_range_for_pressure(
    db: LiquidDatabase,
    liquid_species_full: tuple[str, ...],
    *,
    p_env: float | None,
) -> tuple[float, float]:
    if len(liquid_species_full) == 0:
        raise LiquidDatabaseRangeError("liquid_species_full must not be empty")
    ranges = []
    for name in liquid_species_full:
        rec = get_species_record(db, name)
        if p_env is None:
            ranges.append(rec.cp_T_range)
        else:
            ranges.append(select_pressure_bank(rec, p_env).cp_T_range)
    t_min = max(item[0] for item in ranges)
    t_max = min(item[1] for item in ranges)
    if t_min >= t_max:
        raise LiquidDatabaseRangeError(
            f"liquid species {liquid_species_full} do not share a valid common cp temperature range"
        )
    return (t_min, t_max)


def validate_liquid_species_coverage(db: LiquidDatabase, required_species: tuple[str, ...]) -> None:
    for species_name in required_species:
        resolve_species_name(db, species_name)


def validate_wilke_chang_requirements(db: LiquidDatabase, required_species: tuple[str, ...]) -> None:
    for species_name in required_species:
        record = get_species_record(db, species_name)
        if record.molecular_weight <= 0.0:
            raise LiquidDatabaseValidationError(
                f"{record.name}.molecular_weight must be strictly positive for wilke_chang diffusivity"
            )
        if record.molar_volume <= 0.0:
            raise LiquidDatabaseValidationError(
                f"{record.name}.molar_volume must be strictly positive for wilke_chang diffusivity"
            )
        if record.association_factor <= 0.0:
            raise LiquidDatabaseValidationError(
                f"{record.name}.association_factor must be strictly positive for wilke_chang diffusivity"
            )


def build_liquid_database(run_cfg: RunConfig) -> LiquidDatabase:
    db = load_liquid_database(run_cfg.paths.liquid_database_path)
    validate_liquid_species_coverage(db, run_cfg.species_maps.liq_full_names)
    get_common_cp_temperature_range_for_pressure(db, run_cfg.species_maps.liq_full_names, p_env=run_cfg.pressure)
    return db


def summarize_species_record(rec: LiquidPureSpeciesRecord) -> dict[str, object]:
    return {
        "name": rec.name,
        "aliases": rec.aliases,
        "cp_model": rec.cp_model,
        "cp_T_range": rec.cp_T_range,
        "rho_model": rec.rho_model,
        "k_model": rec.k_model,
        "mu_model": rec.mu_model,
        "activity_model": rec.activity_model,
        "pressure_bank_count": len(rec.pressure_banks),
    }


def summarize_database(db: LiquidDatabase) -> dict[str, object]:
    return {
        "file_type": db.meta.file_type,
        "version": db.meta.version,
        "reference_T": db.meta.reference_T,
        "species_count": len(db.species_by_name),
        "species_names": tuple(db.species_by_name.keys()),
        "has_unifac": db.unifac is not None,
        "liquid_cp_default_model": db.global_models.liquid_cp_default_model,
        "liquid_density_default_model": db.global_models.liquid_density_default_model,
        "activity_model_default": db.global_models.activity_model_default,
    }


__all__ = [
    "LiquidDatabase",
    "LiquidDatabaseError",
    "LiquidDatabaseFormatError",
    "LiquidDatabaseRangeError",
    "LiquidDatabaseValidationError",
    "LiquidDBGlobalModels",
    "LiquidDBMeta",
    "LiquidPureSpeciesRecord",
    "LiquidSpeciesNotFoundError",
    "build_liquid_database",
    "get_common_cp_temperature_range",
    "get_species_record",
    "has_species",
    "load_liquid_database",
    "resolve_species_name",
    "select_pressure_bank",
    "select_pressure_bank_with_diagnostics",
    "summarize_database",
    "summarize_species_record",
    "get_common_cp_temperature_range_for_pressure",
    "validate_liquid_species_coverage",
    "validate_wilke_chang_requirements",
]
