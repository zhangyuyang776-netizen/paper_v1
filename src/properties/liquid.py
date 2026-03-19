from __future__ import annotations

"""Liquid bulk thermo/property backend for Phase 2.

This module intentionally sits between the liquid database / mixture-rule layers
and later recovery / aggregator / interface-physics consumers.

Current Phase 2 v1 keeps the implementation narrow and explicit:
- pure cp model: ``shomate``
- pure density / conductivity / viscosity model: ``merino_log_poly``
- liquid mixture density: ``merino_x_sqrt_rho``
- liquid mixture conductivity: ``filippov``
- liquid mixture viscosity: ``grunberg_nissan``
- liquid diffusivity: ``wilke_chang``
"""

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from properties.liquid_db import (
    LiquidDatabase,
    get_species_record,
    resolve_species_name,
    select_pressure_bank_with_diagnostics,
    validate_wilke_chang_requirements,
)
from properties.mix_rules import (
    MixRulesError,
    mass_to_mole_fractions,
    mixture_liquid_conductivity,
    mixture_liquid_cp,
    mixture_liquid_density,
    mixture_liquid_diffusivity,
    mixture_liquid_enthalpy,
    mixture_liquid_viscosity,
    mixture_molecular_weight_from_mass_fractions,
)

DEFAULT_MIXTURE_DENSITY_MODEL = "merino_x_sqrt_rho"
DEFAULT_MIXTURE_CONDUCTIVITY_MODEL = "filippov"
DEFAULT_MIXTURE_VISCOSITY_MODEL = "grunberg_nissan"
DEFAULT_DIFFUSIVITY_MODEL = "wilke_chang"


class LiquidThermoError(Exception):
    """Base error for liquid thermo/property model construction and evaluation."""


class LiquidThermoValidationError(LiquidThermoError):
    """Raised when liquid thermo inputs or database-backed shapes are invalid."""


class LiquidThermoModelError(LiquidThermoError):
    """Raised when a requested liquid property model is unsupported or inconsistent."""


class LiquidEnthalpyModelError(LiquidThermoError):
    """Raised when the liquid enthalpy model cannot be evaluated robustly."""


@dataclass(frozen=True)
class LiquidResolvedSpeciesRecord:
    name: str
    aliases: tuple[str, ...]
    molecular_weight: float
    molar_volume: float
    association_factor: float
    Tc: float | None
    activity_model: str | None
    unifac_groups: Mapping[str, float] | None
    boiling_temperature_atm: float
    selected_p_fit: float
    boiling_temperature: float
    T_ref: float
    cp_model: str
    cp_T_range: tuple[float, float]
    cp_coeffs: Mapping[str, float]
    hvap_ref: float
    hvap_model: str | None
    hvap_coeffs: Mapping[str, float]
    rho_model: str
    rho_coeffs: Mapping[str, float]
    k_model: str
    k_coeffs: Mapping[str, float]
    mu_model: str
    mu_coeffs: Mapping[str, float]


def _freeze_mapping(value: Mapping[str, float]) -> Mapping[str, float]:
    return MappingProxyType({str(key): float(item) for key, item in value.items()})


def _as_1d_float_array(name: str, value: Any, *, expected_size: int | None = None) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim != 1:
        raise LiquidThermoValidationError(f"{name} must be a one-dimensional float array")
    if arr.size == 0:
        raise LiquidThermoValidationError(f"{name} must be non-empty")
    if not np.all(np.isfinite(arr)):
        raise LiquidThermoValidationError(f"{name} must contain only finite values")
    if expected_size is not None and arr.size != expected_size:
        raise LiquidThermoValidationError(f"{name} must have length {expected_size}")
    return arr


def _validate_temperature(T: float) -> float:
    if isinstance(T, bool) or not np.isscalar(T):
        raise LiquidThermoValidationError("T must be a finite positive scalar")
    T_value = float(T)
    if not np.isfinite(T_value) or T_value <= 0.0:
        raise LiquidThermoValidationError("T must be a finite positive scalar")
    return T_value


def _validate_positive_scalar(name: str, value: float) -> float:
    if isinstance(value, bool) or not np.isscalar(value):
        raise LiquidThermoValidationError(f"{name} must be a finite positive scalar")
    value_float = float(value)
    if not np.isfinite(value_float) or value_float <= 0.0:
        raise LiquidThermoValidationError(f"{name} must be a finite positive scalar")
    return value_float


def _validate_temperature_in_range(T: float, T_range: tuple[float, float]) -> None:
    if T < T_range[0] or T > T_range[1]:
        raise LiquidThermoValidationError(
            f"T={T:.6g} lies outside the valid liquid temperature range [{T_range[0]:.6g}, {T_range[1]:.6g}]"
        )


def _validate_mass_fractions(Y_full: np.ndarray, n_spec: int, *, atol: float = 1.0e-12) -> np.ndarray:
    Y = _as_1d_float_array("Y_full", Y_full, expected_size=n_spec)
    if np.any(Y < 0.0):
        raise LiquidThermoValidationError("Y_full must be non-negative")
    if not np.isclose(float(np.sum(Y)), 1.0, rtol=0.0, atol=atol):
        raise LiquidThermoValidationError("Y_full must sum to 1 within tolerance")
    return Y


def _validate_batch_inputs(T: np.ndarray, Y_full_2d: np.ndarray, *, n_spec: int) -> tuple[np.ndarray, np.ndarray]:
    T_arr = np.asarray(T, dtype=np.float64)
    Y_arr = np.asarray(Y_full_2d, dtype=np.float64)
    if T_arr.ndim != 1:
        raise LiquidThermoValidationError("T batch input must be one-dimensional")
    if Y_arr.ndim != 2:
        raise LiquidThermoValidationError("Y_full_2d must be two-dimensional")
    if Y_arr.shape[0] != T_arr.shape[0]:
        raise LiquidThermoValidationError("Y_full_2d row count must match T length")
    if Y_arr.shape[1] != n_spec:
        raise LiquidThermoValidationError(f"Y_full_2d species dimension must be {n_spec}")
    if not np.all(np.isfinite(T_arr)):
        raise LiquidThermoValidationError("T batch input must contain only finite values")
    if not np.all(np.isfinite(Y_arr)):
        raise LiquidThermoValidationError("Y_full_2d must contain only finite values")
    return T_arr, Y_arr


def _require_supported_model(name: str, model: str, supported: set[str]) -> str:
    if model not in supported:
        raise LiquidThermoModelError(f"Unsupported {name}: {model}")
    return model


def _require_positive_finite_property(name: str, value: float, *, species_name: str, T: float) -> float:
    if not np.isfinite(value) or value <= 0.0:
        raise LiquidThermoModelError(
            f"{name} for species {species_name!r} must be finite and > 0 at T={T:.6g}"
        )
    return float(value)


def _require_finite_enthalpy(value: float, *, species_name: str, T: float) -> float:
    if not np.isfinite(value):
        raise LiquidEnthalpyModelError(f"liquid enthalpy for species {species_name!r} is not finite at T={T:.6g}")
    return float(value)


def _cp_shomate(T: float, coeffs: Mapping[str, float]) -> float:
    t = T / 1000.0
    return (
        coeffs["A"]
        + coeffs["B"] * t
        + coeffs["C"] * t * t
        + coeffs["D"] * t * t * t
        + coeffs["E"] / (t * t)
    )


def _h_shomate_molar(T: float, coeffs: Mapping[str, float]) -> float:
    t = T / 1000.0
    return 1000.0 * (
        coeffs["A"] * t
        + coeffs["B"] * (t * t) / 2.0
        + coeffs["C"] * (t * t * t) / 3.0
        + coeffs["D"] * (t * t * t * t) / 4.0
        - coeffs["E"] / t
        + coeffs["F"]
        - coeffs["H"]
    )


def _eval_merino_log_poly(T: float, coeffs: Mapping[str, float]) -> float:
    log_value = (
        coeffs["A"] * np.log(T)
        + coeffs["B"] / T
        + coeffs["C"] / (T * T)
        + coeffs["D"]
        + coeffs["E"] * T
        + coeffs["F"] * T * T
    )
    return float(np.exp(log_value))


@dataclass(frozen=True)
class LiquidThermoModel:
    db: LiquidDatabase
    species_names: tuple[str, ...]
    species_records: tuple[LiquidResolvedSpeciesRecord, ...]
    molecular_weights: np.ndarray
    association_factors: np.ndarray
    molar_volumes: np.ndarray
    environment_pressure: float
    selected_bank_pressures: np.ndarray
    reference_T: float
    density_model: str
    cp_model: str
    conductivity_model: str
    viscosity_model: str
    diffusivity_model: str
    mixture_density_model: str
    mixture_conductivity_model: str
    mixture_viscosity_model: str

    def __post_init__(self) -> None:
        if len(self.species_names) == 0:
            raise LiquidThermoValidationError("species_names must be non-empty")
        if len(set(self.species_names)) != len(self.species_names):
            raise LiquidThermoValidationError("species_names must not contain duplicates")
        if len(self.species_records) != len(self.species_names):
            raise LiquidThermoValidationError("species_records length must match species_names")
        for name, record in zip(self.species_names, self.species_records, strict=True):
            if record.name != name:
                raise LiquidThermoValidationError("species_records must be aligned with species_names")

        mw = _as_1d_float_array("molecular_weights", self.molecular_weights, expected_size=len(self.species_names))
        if np.any(mw <= 0.0):
            raise LiquidThermoValidationError("molecular_weights must be strictly positive")
        mw.setflags(write=False)
        object.__setattr__(self, "molecular_weights", mw)

        association_factors = _as_1d_float_array(
            "association_factors",
            self.association_factors,
            expected_size=len(self.species_names),
        )
        if np.any(association_factors <= 0.0):
            raise LiquidThermoValidationError("association_factors must be strictly positive")
        association_factors.setflags(write=False)
        object.__setattr__(self, "association_factors", association_factors)

        molar_volumes = _as_1d_float_array(
            "molar_volumes",
            self.molar_volumes,
            expected_size=len(self.species_names),
        )
        if np.any(molar_volumes <= 0.0):
            raise LiquidThermoValidationError("molar_volumes must be strictly positive")
        molar_volumes.setflags(write=False)
        object.__setattr__(self, "molar_volumes", molar_volumes)

        ref_T = _validate_temperature(self.reference_T)
        object.__setattr__(self, "reference_T", ref_T)
        env_p = _validate_positive_scalar("environment_pressure", self.environment_pressure)
        object.__setattr__(self, "environment_pressure", env_p)

        selected_bank_pressures = _as_1d_float_array(
            "selected_bank_pressures",
            self.selected_bank_pressures,
            expected_size=len(self.species_names),
        )
        if np.any(selected_bank_pressures <= 0.0):
            raise LiquidThermoValidationError("selected_bank_pressures must be strictly positive")
        selected_bank_pressures.setflags(write=False)
        object.__setattr__(self, "selected_bank_pressures", selected_bank_pressures)

        _require_supported_model("cp_model", self.cp_model, {"shomate"})
        _require_supported_model("density_model", self.density_model, {"merino_log_poly"})
        _require_supported_model("conductivity_model", self.conductivity_model, {"merino_log_poly"})
        _require_supported_model("viscosity_model", self.viscosity_model, {"merino_log_poly"})
        _require_supported_model(
            "diffusivity_model",
            self.diffusivity_model,
            {"wilke_chang"},
        )
        _require_supported_model(
            "mixture_density_model",
            self.mixture_density_model,
            {"merino_x_sqrt_rho"},
        )
        _require_supported_model(
            "mixture_conductivity_model",
            self.mixture_conductivity_model,
            {"filippov"},
        )
        _require_supported_model(
            "mixture_viscosity_model",
            self.mixture_viscosity_model,
            {"grunberg_nissan"},
        )

    @property
    def n_species(self) -> int:
        return len(self.species_names)

    def valid_temperature_range(self, species_subset: tuple[str, ...] | None = None) -> tuple[float, float]:
        subset = self.species_names if species_subset is None else tuple(species_subset)
        if len(subset) == 0:
            raise LiquidThermoValidationError("species_subset must be non-empty")
        if not set(subset).issubset(set(self.species_names)):
            raise LiquidThermoValidationError("species_subset must be a subset of this liquid thermo model species")
        ranges = [self.species_records[self.species_names.index(name)].cp_T_range for name in subset]
        t_min = max(item[0] for item in ranges)
        t_max = min(item[1] for item in ranges)
        if t_min >= t_max:
            raise LiquidThermoValidationError("species_subset does not share a valid common cp temperature range")
        return (t_min, t_max)

    def mole_fractions(self, Y_full: np.ndarray) -> np.ndarray:
        Y = _validate_mass_fractions(Y_full, self.n_species)
        return mass_to_mole_fractions(Y, self.molecular_weights)

    def mixture_molecular_weight(self, Y_full: np.ndarray) -> float:
        Y = _validate_mass_fractions(Y_full, self.n_species)
        return mixture_molecular_weight_from_mass_fractions(Y, self.molecular_weights)

    def _pure_cp(self, T: float, rec: LiquidPureSpeciesRecord) -> float:
        if rec.cp_model != "shomate":
            raise LiquidThermoModelError(f"Unsupported cp_model for species {rec.name!r}: {rec.cp_model}")
        cp_mass = _cp_shomate(T, _freeze_mapping(rec.cp_coeffs)) / rec.molecular_weight
        return _require_positive_finite_property("liquid cp", cp_mass, species_name=rec.name, T=T)

    def _pure_enthalpy(self, T: float, rec: LiquidPureSpeciesRecord) -> float:
        if rec.cp_model != "shomate":
            raise LiquidThermoModelError(f"Unsupported cp_model for species {rec.name!r}: {rec.cp_model}")
        cp_coeffs = _freeze_mapping(rec.cp_coeffs)
        h_mass = (_h_shomate_molar(T, cp_coeffs) - _h_shomate_molar(rec.T_ref, cp_coeffs)) / rec.molecular_weight
        return _require_finite_enthalpy(h_mass, species_name=rec.name, T=T)

    def _pure_density(self, T: float, rec: LiquidPureSpeciesRecord) -> float:
        if rec.rho_model != "merino_log_poly":
            raise LiquidThermoModelError(f"Unsupported rho_model for species {rec.name!r}: {rec.rho_model}")
        rho = _eval_merino_log_poly(T, _freeze_mapping(rec.rho_coeffs))
        return _require_positive_finite_property("liquid density", rho, species_name=rec.name, T=T)

    def _pure_conductivity(self, T: float, rec: LiquidPureSpeciesRecord) -> float:
        if rec.k_model != "merino_log_poly":
            raise LiquidThermoModelError(f"Unsupported k_model for species {rec.name!r}: {rec.k_model}")
        k = _eval_merino_log_poly(T, _freeze_mapping(rec.k_coeffs))
        return _require_positive_finite_property("liquid conductivity", k, species_name=rec.name, T=T)

    def _pure_viscosity(self, T: float, rec: LiquidPureSpeciesRecord) -> float:
        if rec.mu_model != "merino_log_poly":
            raise LiquidThermoModelError(f"Unsupported mu_model for species {rec.name!r}: {rec.mu_model}")
        mu = _eval_merino_log_poly(T, _freeze_mapping(rec.mu_coeffs))
        return _require_positive_finite_property("liquid viscosity", mu, species_name=rec.name, T=T)

    def _validated_temperature(self, T: float) -> float:
        T_value = _validate_temperature(T)
        _validate_temperature_in_range(T_value, self.valid_temperature_range())
        return T_value

    def pure_density_vector(self, T: float) -> np.ndarray:
        T_value = self._validated_temperature(T)
        return np.array([self._pure_density(T_value, rec) for rec in self.species_records], dtype=np.float64)

    def pure_cp_vector(self, T: float) -> np.ndarray:
        T_value = self._validated_temperature(T)
        return np.array([self._pure_cp(T_value, rec) for rec in self.species_records], dtype=np.float64)

    def pure_enthalpy_vector(self, T: float) -> np.ndarray:
        T_value = self._validated_temperature(T)
        return np.array([self._pure_enthalpy(T_value, rec) for rec in self.species_records], dtype=np.float64)

    def pure_conductivity_vector(self, T: float) -> np.ndarray:
        T_value = self._validated_temperature(T)
        return np.array([self._pure_conductivity(T_value, rec) for rec in self.species_records], dtype=np.float64)

    def pure_viscosity_vector(self, T: float) -> np.ndarray:
        T_value = self._validated_temperature(T)
        return np.array([self._pure_viscosity(T_value, rec) for rec in self.species_records], dtype=np.float64)

    def density_mass(self, T: float, Y_full: np.ndarray) -> float:
        T_value = self._validated_temperature(T)
        Y = _validate_mass_fractions(Y_full, self.n_species)
        X = self.mole_fractions(Y)
        rho_pure = self.pure_density_vector(T_value)
        try:
            return mixture_liquid_density(
                Y=Y,
                X=X,
                rho_pure=rho_pure,
                model=self.mixture_density_model,
            )
        except MixRulesError as exc:
            raise LiquidThermoModelError(str(exc)) from exc

    def cp_mass(self, T: float, Y_full: np.ndarray) -> float:
        T_value = self._validated_temperature(T)
        Y = _validate_mass_fractions(Y_full, self.n_species)
        cp_pure = self.pure_cp_vector(T_value)
        try:
            return mixture_liquid_cp(Y=Y, cp_pure=cp_pure, model="mass_weighted")
        except MixRulesError as exc:
            raise LiquidThermoModelError(str(exc)) from exc

    def enthalpy_mass(self, T: float, Y_full: np.ndarray) -> float:
        T_value = self._validated_temperature(T)
        Y = _validate_mass_fractions(Y_full, self.n_species)
        h_pure = self.pure_enthalpy_vector(T_value)
        try:
            return mixture_liquid_enthalpy(Y=Y, h_pure=h_pure, model="mass_weighted")
        except MixRulesError as exc:
            raise LiquidEnthalpyModelError(str(exc)) from exc

    def conductivity(self, T: float, Y_full: np.ndarray) -> float:
        T_value = self._validated_temperature(T)
        Y = _validate_mass_fractions(Y_full, self.n_species)
        X = self.mole_fractions(Y)
        k_pure = self.pure_conductivity_vector(T_value)
        try:
            return mixture_liquid_conductivity(
                Y=Y,
                X=X,
                k_pure=k_pure,
                model=self.mixture_conductivity_model,
            )
        except MixRulesError as exc:
            raise LiquidThermoModelError(str(exc)) from exc

    def viscosity(self, T: float, Y_full: np.ndarray) -> float:
        T_value = self._validated_temperature(T)
        Y = _validate_mass_fractions(Y_full, self.n_species)
        X = self.mole_fractions(Y)
        mu_pure = self.pure_viscosity_vector(T_value)
        try:
            return mixture_liquid_viscosity(
                Y=Y,
                X=X,
                mu_pure=mu_pure,
                model=self.mixture_viscosity_model,
            )
        except MixRulesError as exc:
            raise LiquidThermoModelError(str(exc)) from exc

    def diffusivity(self, T: float, Y_full: np.ndarray) -> np.ndarray | None:
        T_value = self._validated_temperature(T)
        Y = _validate_mass_fractions(Y_full, self.n_species)
        if self.n_species == 1:
            return None
        X = self.mole_fractions(Y)
        mu_mix = self.viscosity(T_value, Y)
        try:
            return mixture_liquid_diffusivity(
                Y=Y,
                X=X,
                model=self.diffusivity_model,
                T=T_value,
                mu_mix=mu_mix,
                molecular_weights=self.molecular_weights,
                association_factors=self.association_factors,
                molar_volumes=self.molar_volumes,
            )
        except MixRulesError as exc:
            raise LiquidThermoModelError(str(exc)) from exc

    def density_mass_batch(self, T: np.ndarray, Y_full_2d: np.ndarray) -> np.ndarray:
        T_arr, Y_arr = _validate_batch_inputs(T, Y_full_2d, n_spec=self.n_species)
        return np.array(
            [self.density_mass(float(T_i), Y_arr[i, :]) for i, T_i in enumerate(T_arr)],
            dtype=np.float64,
        )

    def cp_mass_batch(self, T: np.ndarray, Y_full_2d: np.ndarray) -> np.ndarray:
        T_arr, Y_arr = _validate_batch_inputs(T, Y_full_2d, n_spec=self.n_species)
        return np.array(
            [self.cp_mass(float(T_i), Y_arr[i, :]) for i, T_i in enumerate(T_arr)],
            dtype=np.float64,
        )

    def enthalpy_mass_batch(self, T: np.ndarray, Y_full_2d: np.ndarray) -> np.ndarray:
        T_arr, Y_arr = _validate_batch_inputs(T, Y_full_2d, n_spec=self.n_species)
        return np.array(
            [self.enthalpy_mass(float(T_i), Y_arr[i, :]) for i, T_i in enumerate(T_arr)],
            dtype=np.float64,
        )


def build_liquid_thermo_model(
    *,
    liquid_db: LiquidDatabase,
    liquid_species_full: tuple[str, ...],
    molecular_weights: np.ndarray,
    p_env: float,
    mixture_density_model: str | None = None,
    mixture_conductivity_model: str | None = None,
    mixture_viscosity_model: str | None = None,
    diffusivity_model: str | None = None,
) -> LiquidThermoModel:
    if len(liquid_species_full) == 0:
        raise LiquidThermoValidationError("liquid_species_full must be non-empty")
    canonical_species = tuple(resolve_species_name(liquid_db, name) for name in liquid_species_full)
    if len(set(canonical_species)) != len(canonical_species):
        raise LiquidThermoValidationError("liquid_species_full must resolve to unique canonical species names")

    base_records = tuple(get_species_record(liquid_db, name) for name in canonical_species)
    mw = _as_1d_float_array("molecular_weights", molecular_weights, expected_size=len(canonical_species))
    expected_mw = np.array([record.molecular_weight for record in base_records], dtype=np.float64)
    if not np.allclose(mw, expected_mw, rtol=1.0e-10, atol=1.0e-12):
        raise LiquidThermoValidationError("molecular_weights must match the database species records")

    cp_model = liquid_db.global_models.liquid_cp_default_model
    density_model = liquid_db.global_models.liquid_density_default_model
    conductivity_model = liquid_db.global_models.liquid_conductivity_default_model
    viscosity_model = liquid_db.global_models.liquid_viscosity_default_model
    mixture_density_model_resolved = (
        liquid_db.global_models.liquid_mixture_density_model
        if mixture_density_model is None
        else mixture_density_model
    )
    mixture_conductivity_model_resolved = (
        liquid_db.global_models.liquid_mixture_conductivity_model
        if mixture_conductivity_model is None
        else mixture_conductivity_model
    )
    mixture_viscosity_model_resolved = (
        liquid_db.global_models.liquid_mixture_viscosity_model
        if mixture_viscosity_model is None
        else mixture_viscosity_model
    )
    diffusivity_model_resolved = (
        liquid_db.global_models.liquid_diffusion_model
        if diffusivity_model is None
        else diffusivity_model
    )
    if diffusivity_model_resolved == "wilke_chang":
        validate_wilke_chang_requirements(liquid_db, canonical_species)

    resolved_records_list: list[LiquidResolvedSpeciesRecord] = []
    selected_bank_pressures: list[float] = []
    reference_T: float | None = None
    for record in base_records:
        selected_bank, _ = select_pressure_bank_with_diagnostics(record, p_env)
        if reference_T is None:
            reference_T = selected_bank.T_ref
        elif not np.isclose(selected_bank.T_ref, reference_T, rtol=0.0, atol=1.0e-12):
            raise LiquidThermoValidationError("all selected liquid pressure banks must share the same T_ref")
        if selected_bank.cp_model != cp_model:
            raise LiquidThermoModelError(
                f"Species {record.name!r} cp_model={selected_bank.cp_model} does not match global default {cp_model}"
            )
        if selected_bank.rho_model != density_model:
            raise LiquidThermoModelError(
                f"Species {record.name!r} rho_model={selected_bank.rho_model} does not match global default {density_model}"
            )
        if selected_bank.k_model != conductivity_model:
            raise LiquidThermoModelError(
                f"Species {record.name!r} k_model={selected_bank.k_model} does not match global default {conductivity_model}"
            )
        if selected_bank.mu_model != viscosity_model:
            raise LiquidThermoModelError(
                f"Species {record.name!r} mu_model={selected_bank.mu_model} does not match global default {viscosity_model}"
            )
        resolved_records_list.append(
            LiquidResolvedSpeciesRecord(
                name=record.name,
                aliases=record.aliases,
                molecular_weight=record.molecular_weight,
                molar_volume=record.molar_volume,
                association_factor=record.association_factor,
                Tc=(record.Tc if record.Tc is not None else selected_bank.Tc),
                activity_model=record.activity_model,
                unifac_groups=record.unifac_groups,
                boiling_temperature_atm=(
                    record.boiling_temperature
                    if record.boiling_temperature_atm is None
                    else record.boiling_temperature_atm
                ),
                selected_p_fit=selected_bank.p_fit,
                boiling_temperature=selected_bank.boiling_temperature,
                T_ref=selected_bank.T_ref,
                cp_model=selected_bank.cp_model,
                cp_T_range=selected_bank.cp_T_range,
                cp_coeffs=selected_bank.cp_coeffs,
                hvap_ref=selected_bank.hvap_ref,
                hvap_model=selected_bank.hvap_model,
                hvap_coeffs=selected_bank.hvap_coeffs,
                rho_model=selected_bank.rho_model,
                rho_coeffs=selected_bank.rho_coeffs,
                k_model=selected_bank.k_model,
                k_coeffs=selected_bank.k_coeffs,
                mu_model=selected_bank.mu_model,
                mu_coeffs=selected_bank.mu_coeffs,
            )
        )
        selected_bank_pressures.append(selected_bank.p_fit)

    assert reference_T is not None
    resolved_records = tuple(resolved_records_list)

    return LiquidThermoModel(
        db=liquid_db,
        species_names=canonical_species,
        species_records=resolved_records,
        molecular_weights=mw.copy(),
        association_factors=np.array([record.association_factor for record in resolved_records], dtype=np.float64),
        molar_volumes=np.array([record.molar_volume for record in resolved_records], dtype=np.float64),
        environment_pressure=float(p_env),
        selected_bank_pressures=np.array(selected_bank_pressures, dtype=np.float64),
        reference_T=reference_T,
        density_model=density_model,
        cp_model=cp_model,
        conductivity_model=conductivity_model,
        viscosity_model=viscosity_model,
        diffusivity_model=diffusivity_model_resolved,
        mixture_density_model=mixture_density_model_resolved,
        mixture_conductivity_model=mixture_conductivity_model_resolved,
        mixture_viscosity_model=mixture_viscosity_model_resolved,
    )


__all__ = [
    "DEFAULT_DIFFUSIVITY_MODEL",
    "DEFAULT_MIXTURE_CONDUCTIVITY_MODEL",
    "DEFAULT_MIXTURE_DENSITY_MODEL",
    "DEFAULT_MIXTURE_VISCOSITY_MODEL",
    "LiquidEnthalpyModelError",
    "LiquidThermoError",
    "LiquidThermoModel",
    "LiquidThermoModelError",
    "LiquidThermoValidationError",
    "build_liquid_thermo_model",
]
