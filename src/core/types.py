from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt


FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]
PathLike = str | Path


def _require_nonempty(name: str, value: Any) -> None:
    if isinstance(value, np.ndarray):
        if value.size == 0:
            raise ValueError(f"{name} must be non-empty")
        return
    if len(value) == 0:  # type: ignore[arg-type]
        raise ValueError(f"{name} must be non-empty")


def _as_float_array(name: str, value: Any, ndim: int | None = None) -> FloatArray:
    try:
        arr = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be convertible to a float64 ndarray") from exc
    if ndim is not None and arr.ndim != ndim:
        raise ValueError(f"{name} must have ndim={ndim}, got {arr.ndim}")
    return arr


def _as_int_array(name: str, value: Any, ndim: int | None = None) -> IntArray:
    raw = np.asarray(value)
    if not np.issubdtype(raw.dtype, np.integer):
        raise TypeError(f"{name} must contain integers")
    arr = np.asarray(value, dtype=np.int64)
    if ndim is not None and arr.ndim != ndim:
        raise ValueError(f"{name} must have ndim={ndim}, got {arr.ndim}")
    return arr


def _check_same_length(name_a: str, a: Any, name_b: str, b: Any) -> None:
    if len(a) != len(b):  # type: ignore[arg-type]
        raise ValueError(f"{name_a} and {name_b} must have the same length")


def _check_all_finite(name: str, arr: FloatArray) -> None:
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")


def _check_array_row_count(name: str, arr: FloatArray | IntArray, expected: int) -> None:
    if arr.shape[0] != expected:
        raise ValueError(f"{name} first dimension must be {expected}, got {arr.shape[0]}")


def _check_positive(name: str, value: float) -> None:
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be a positive finite value")


def _check_nonnegative(name: str, value: float) -> None:
    if not np.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be a non-negative finite value")


def _check_nonnegative_int(name: str, value: int) -> None:
    if value < 0:
        raise ValueError(f"{name} must be >= 0")


def _check_strictly_increasing(name: str, arr: FloatArray) -> None:
    if np.any(np.diff(arr) <= 0.0):
        raise ValueError(f"{name} must be strictly increasing")


def _validate_name_tuple(name: str, values: tuple[str, ...], *, allow_empty: bool) -> None:
    if not allow_empty:
        _require_nonempty(name, values)
    if len(set(values)) != len(values):
        raise ValueError(f"{name} must not contain duplicates")
    for item in values:
        if not item:
            raise ValueError(f"{name} must not contain empty strings")


def _slice_start(s: slice) -> int:
    return 0 if s.start is None else int(s.start)


def _slice_stop(s: slice) -> int:
    if s.stop is None:
        raise ValueError("slice stop must not be None")
    return int(s.stop)


@dataclass(slots=True, kw_only=True, frozen=True)
class CasePaths:
    """Resolved filesystem paths for a single case."""

    config_path: PathLike
    case_root: PathLike
    gas_mechanism_path: PathLike
    liquid_database_path: PathLike
    output_root: PathLike
    normalized_config_path: PathLike | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "config_path", Path(self.config_path))
        object.__setattr__(self, "case_root", Path(self.case_root))
        object.__setattr__(self, "gas_mechanism_path", Path(self.gas_mechanism_path))
        object.__setattr__(self, "liquid_database_path", Path(self.liquid_database_path))
        object.__setattr__(self, "output_root", Path(self.output_root))
        if self.normalized_config_path is not None:
            object.__setattr__(self, "normalized_config_path", Path(self.normalized_config_path))

    @property
    def mechanism_path(self) -> Path:
        return self.gas_mechanism_path

    @property
    def species_database_path(self) -> Path:
        return self.liquid_database_path


@dataclass(slots=True, kw_only=True, frozen=True)
class MeshConfig:
    """Normalized mesh/build parameters from the raw mesh section."""

    a0: float
    r_end: float
    n_liq: int
    n_gas_near: int
    far_stretch_ratio: float

    def __post_init__(self) -> None:
        _check_positive("a0", self.a0)
        if not np.isfinite(self.r_end) or self.r_end <= self.a0:
            raise ValueError("r_end must be finite and greater than a0")
        if self.n_liq < 1:
            raise ValueError("n_liq must be >= 1")
        if self.n_gas_near < 1:
            raise ValueError("n_gas_near must be >= 1")
        if not np.isfinite(self.far_stretch_ratio) or self.far_stretch_ratio < 1.0:
            raise ValueError("far_stretch_ratio must be finite and >= 1")


@dataclass(slots=True, kw_only=True, frozen=True)
class SpeciesControlConfig:
    """Normalized control/config metadata for species bookkeeping."""

    gas_closure_species: str
    liquid_closure_species: str | None
    liquid_to_gas_species_map: dict[str, str]

    def __post_init__(self) -> None:
        if not self.gas_closure_species:
            raise ValueError("gas_closure_species must be non-empty")
        if self.liquid_closure_species is not None and not self.liquid_closure_species:
            raise ValueError("liquid_closure_species must be non-empty when provided")
        copied_map = dict(self.liquid_to_gas_species_map)
        if len(copied_map) == 0:
            raise ValueError("liquid_to_gas_species_map must be non-empty")
        for liq_name, gas_name in copied_map.items():
            if not liq_name:
                raise ValueError("liquid_to_gas_species_map keys must be non-empty")
            if not gas_name:
                raise ValueError("liquid_to_gas_species_map values must be non-empty")
        object.__setattr__(self, "liquid_to_gas_species_map", copied_map)

    @property
    def mapped_vapor_gas_names(self) -> tuple[str, ...]:
        return tuple(self.liquid_to_gas_species_map.values())


@dataclass(slots=True, kw_only=True, frozen=True)
class InitializationConfig:
    """Normalized full-order initialization vectors and scalar state."""

    gas_temperature: float
    gas_pressure: float
    liquid_temperature: float
    gas_y_full_0: FloatArray
    liquid_y_full_0: FloatArray
    y_vap_if0_gas_full: FloatArray
    t_init_T: float

    def __post_init__(self) -> None:
        _check_positive("gas_temperature", self.gas_temperature)
        _check_positive("gas_pressure", self.gas_pressure)
        _check_positive("liquid_temperature", self.liquid_temperature)
        _check_positive("t_init_T", self.t_init_T)

        gas_y_full_0 = _as_float_array("gas_y_full_0", self.gas_y_full_0, ndim=1)
        liquid_y_full_0 = _as_float_array("liquid_y_full_0", self.liquid_y_full_0, ndim=1)
        y_vap_if0_gas_full = _as_float_array("y_vap_if0_gas_full", self.y_vap_if0_gas_full, ndim=1)
        _require_nonempty("gas_y_full_0", gas_y_full_0)
        _require_nonempty("liquid_y_full_0", liquid_y_full_0)
        _require_nonempty("y_vap_if0_gas_full", y_vap_if0_gas_full)
        _check_all_finite("gas_y_full_0", gas_y_full_0)
        _check_all_finite("liquid_y_full_0", liquid_y_full_0)
        _check_all_finite("y_vap_if0_gas_full", y_vap_if0_gas_full)
        object.__setattr__(self, "gas_y_full_0", gas_y_full_0)
        object.__setattr__(self, "liquid_y_full_0", liquid_y_full_0)
        object.__setattr__(self, "y_vap_if0_gas_full", y_vap_if0_gas_full)


@dataclass(slots=True, kw_only=True, frozen=True)
class TimeStepperConfig:
    """Normalized step-size control parameters."""

    t0: float
    t_end: float
    dt_start: float
    dt_min: float
    dt_max: float
    max_retries_per_step: int
    accept_growth_factor: float
    reject_shrink_factor: float

    def __post_init__(self) -> None:
        _check_positive("dt_min", self.dt_min)
        _check_positive("dt_start", self.dt_start)
        _check_positive("dt_max", self.dt_max)
        if not (self.dt_min < self.dt_start <= self.dt_max):
            raise ValueError("Require dt_min < dt_start <= dt_max")
        if not np.isfinite(self.t0) or not np.isfinite(self.t_end) or self.t_end <= self.t0:
            raise ValueError("Require finite t0 and t_end > t0")
        _check_nonnegative_int("max_retries_per_step", self.max_retries_per_step)
        if not np.isfinite(self.accept_growth_factor) or self.accept_growth_factor < 1.0:
            raise ValueError("accept_growth_factor must be >= 1")
        if not np.isfinite(self.reject_shrink_factor) or not (0.0 < self.reject_shrink_factor < 1.0):
            raise ValueError("reject_shrink_factor must satisfy 0 < value < 1")


@dataclass(slots=True, kw_only=True, frozen=True)
class OuterStepperConfig:
    """Outer predictor-corrector control parameters."""

    outer_max_iter: int
    predictor_mode: str
    corrector_mode: str
    omega_a: float
    omega_v: float
    outer_convergence_mode: str
    outer_convergence_tol: float
    eps_ref_dot_a: float

    def __post_init__(self) -> None:
        if self.outer_max_iter < 1:
            raise ValueError("outer_max_iter must be >= 1")
        if self.predictor_mode != "explicit_from_previous_dot_a":
            raise ValueError("predictor_mode must be 'explicit_from_previous_dot_a'")
        if self.corrector_mode != "trapezoidal_fixed_point":
            raise ValueError("corrector_mode must be 'trapezoidal_fixed_point'")
        if self.outer_convergence_mode != "eps_dot_a":
            raise ValueError("outer_convergence_mode must be 'eps_dot_a'")
        _check_positive("omega_a", self.omega_a)
        _check_positive("omega_v", self.omega_v)
        if self.omega_a > 1.0:
            raise ValueError("omega_a must be <= 1")
        if self.omega_v > 1.0:
            raise ValueError("omega_v must be <= 1")
        _check_positive("outer_convergence_tol", self.outer_convergence_tol)
        _check_positive("eps_ref_dot_a", self.eps_ref_dot_a)


@dataclass(slots=True, kw_only=True, frozen=True)
class FieldSplitBulkConfig:
    """Normalized bulk fieldsplit sub-KSP configuration."""

    ksp_type: str
    pc_type: str
    sub_ksp_type: str
    sub_pc_type: str
    asm_overlap: int

    def __post_init__(self) -> None:
        if not self.ksp_type:
            raise ValueError("fieldsplit.bulk.ksp_type must be non-empty")
        if not self.pc_type:
            raise ValueError("fieldsplit.bulk.pc_type must be non-empty")
        if not self.sub_ksp_type:
            raise ValueError("fieldsplit.bulk.sub_ksp_type must be non-empty")
        if not self.sub_pc_type:
            raise ValueError("fieldsplit.bulk.sub_pc_type must be non-empty")
        _check_nonnegative_int("fieldsplit.bulk.asm_overlap", self.asm_overlap)


@dataclass(slots=True, kw_only=True, frozen=True)
class FieldSplitIfaceConfig:
    """Normalized interface fieldsplit sub-KSP configuration."""

    ksp_type: str
    pc_type: str

    def __post_init__(self) -> None:
        if not self.ksp_type:
            raise ValueError("fieldsplit.iface.ksp_type must be non-empty")
        if not self.pc_type:
            raise ValueError("fieldsplit.iface.pc_type must be non-empty")


@dataclass(slots=True, kw_only=True, frozen=True)
class FieldSplitConfig:
    """Normalized fieldsplit / Schur configuration."""

    scheme: str
    type: str
    schur_fact_type: str
    schur_precondition: str
    bulk: FieldSplitBulkConfig
    iface: FieldSplitIfaceConfig

    def __post_init__(self) -> None:
        if self.scheme != "bulk_iface":
            raise ValueError("fieldsplit.scheme must be 'bulk_iface'")
        if self.type not in {"schur", "additive", "multiplicative", "symmetric_multiplicative"}:
            raise ValueError("fieldsplit.type is unsupported")
        if self.schur_fact_type != "full":
            raise ValueError("fieldsplit.schur_fact_type must be 'full'")
        if self.schur_precondition != "a11":
            raise ValueError("fieldsplit.schur_precondition must be 'a11'")


@dataclass(slots=True, kw_only=True, frozen=True)
class InnerSolverConfig:
    """Normalized inner nonlinear solver settings."""

    snes_type: str
    linesearch_type: str
    snes_rtol: float
    snes_atol: float
    snes_stol: float
    snes_max_it: int
    options_prefix: str
    lag_jacobian: int
    lag_preconditioner: int

    ksp_type: str
    pc_type: str
    ksp_rtol: float
    ksp_atol: float
    ksp_max_it: int
    restart: int
    gmres_modified_gram_schmidt: bool
    gmres_preallocate: bool

    fieldsplit: FieldSplitConfig

    def __post_init__(self) -> None:
        if self.snes_type != "newtonls":
            raise ValueError("snes_type must be 'newtonls'")
        if self.linesearch_type != "bt":
            raise ValueError("linesearch_type must be 'bt'")
        _check_nonnegative("snes_rtol", self.snes_rtol)
        _check_nonnegative("snes_atol", self.snes_atol)
        _check_nonnegative("snes_stol", self.snes_stol)
        if self.snes_max_it < 1:
            raise ValueError("snes_max_it must be >= 1")
        if self.lag_jacobian < -1:
            raise ValueError("lag_jacobian must be >= -1")
        if self.lag_preconditioner < -1:
            raise ValueError("lag_preconditioner must be >= -1")
        if self.ksp_type not in {"fgmres", "gmres", "preonly"}:
            raise ValueError("ksp_type is unsupported")
        if self.pc_type not in {"fieldsplit", "asm", "ilu", "lu", "jacobi", "none"}:
            raise ValueError("pc_type is unsupported")
        _check_nonnegative("ksp_rtol", self.ksp_rtol)
        _check_nonnegative("ksp_atol", self.ksp_atol)
        if self.ksp_max_it < 1:
            raise ValueError("ksp_max_it must be >= 1")
        if self.restart < 1:
            raise ValueError("restart must be >= 1")
        if not self.pc_type:
            raise ValueError("pc_type must be non-empty")
        if not self.ksp_type:
            raise ValueError("ksp_type must be non-empty")


@dataclass(slots=True, kw_only=True, frozen=True)
class RecoveryConfig:
    """Bounds and tolerances for state recovery / enthalpy inversion."""

    T_min_l: float
    T_max_l: float
    T_min_g: float
    T_max_g: float
    liq_h_inv_tol: float
    liq_h_inv_max_iter: int
    gas_h_inv_tol: float
    gas_h_inv_max_iter: int
    use_cantera_hpy_first: bool

    def __post_init__(self) -> None:
        if self.T_max_l <= self.T_min_l:
            raise ValueError("Require T_max_l > T_min_l")
        if self.T_max_g <= self.T_min_g:
            raise ValueError("Require T_max_g > T_min_g")
        _check_positive("liq_h_inv_tol", self.liq_h_inv_tol)
        _check_positive("gas_h_inv_tol", self.gas_h_inv_tol)
        if self.liq_h_inv_max_iter < 1:
            raise ValueError("liq_h_inv_max_iter must be >= 1")
        if self.gas_h_inv_max_iter < 1:
            raise ValueError("gas_h_inv_max_iter must be >= 1")


@dataclass(slots=True, kw_only=True, frozen=True)
class DiagnosticsConfig:
    """Runtime logging and diagnostics switches."""

    verbose_interface_panel: bool
    verbose_property_warnings: bool
    write_step_diag: bool
    write_interface_diag: bool
    write_failure_report: bool
    output_every_n_steps: int

    def __post_init__(self) -> None:
        if self.output_every_n_steps < 1:
            raise ValueError("output_every_n_steps must be >= 1")


@dataclass(slots=True, kw_only=True, frozen=True)
class OutputConfig:
    """Output switches for fields and timeseries."""

    write_spatial_fields: bool
    write_spatial_species: bool
    write_time_series_scalars: bool
    write_time_series_species: bool
    snapshot_format: str

    def __post_init__(self) -> None:
        if self.snapshot_format != "npz":
            raise ValueError("snapshot_format must be 'npz'")


@dataclass(slots=True, kw_only=True, frozen=True)
class ValidationConfig:
    """Runtime validation and post-step assertions."""

    enable_mass_balance_check: bool
    enable_energy_balance_check: bool
    enable_state_bounds_check: bool


@dataclass(slots=True, kw_only=True, frozen=True)
class SpeciesMaps:
    """Full/reduced species bookkeeping shared across modules."""

    liq_full_names: tuple[str, ...]
    liq_active_names: tuple[str, ...]
    liq_closure_name: str | None
    gas_full_names: tuple[str, ...]
    gas_active_names: tuple[str, ...]
    gas_closure_name: str
    liq_full_to_reduced: IntArray
    liq_reduced_to_full: IntArray
    gas_full_to_reduced: IntArray
    gas_reduced_to_full: IntArray
    liq_full_to_gas_full: IntArray

    def __post_init__(self) -> None:
        _validate_name_tuple("liq_full_names", self.liq_full_names, allow_empty=False)
        _validate_name_tuple("liq_active_names", self.liq_active_names, allow_empty=True)
        _validate_name_tuple("gas_full_names", self.gas_full_names, allow_empty=False)
        _validate_name_tuple("gas_active_names", self.gas_active_names, allow_empty=False)

        liq_full_to_reduced = _as_int_array("liq_full_to_reduced", self.liq_full_to_reduced, ndim=1)
        liq_reduced_to_full = _as_int_array("liq_reduced_to_full", self.liq_reduced_to_full, ndim=1)
        gas_full_to_reduced = _as_int_array("gas_full_to_reduced", self.gas_full_to_reduced, ndim=1)
        gas_reduced_to_full = _as_int_array("gas_reduced_to_full", self.gas_reduced_to_full, ndim=1)
        liq_full_to_gas_full = _as_int_array("liq_full_to_gas_full", self.liq_full_to_gas_full, ndim=1)
        object.__setattr__(self, "liq_full_to_reduced", liq_full_to_reduced)
        object.__setattr__(self, "liq_reduced_to_full", liq_reduced_to_full)
        object.__setattr__(self, "gas_full_to_reduced", gas_full_to_reduced)
        object.__setattr__(self, "gas_reduced_to_full", gas_reduced_to_full)
        object.__setattr__(self, "liq_full_to_gas_full", liq_full_to_gas_full)
        liq_full_to_reduced.setflags(write=False)
        liq_reduced_to_full.setflags(write=False)
        gas_full_to_reduced.setflags(write=False)
        gas_reduced_to_full.setflags(write=False)
        liq_full_to_gas_full.setflags(write=False)

        if len(self.liq_full_names) == 1:
            if self.liq_closure_name is not None and self.liq_closure_name not in self.liq_full_names:
                raise ValueError("liq_closure_name must be a liquid full species when provided")
        else:
            if self.liq_closure_name is None:
                raise ValueError("liq_closure_name must be provided for multicomponent liquid")
            if self.liq_closure_name not in self.liq_full_names:
                raise ValueError("liq_closure_name must be in liq_full_names")

        if not self.gas_closure_name:
            raise ValueError("gas_closure_name must be non-empty")
        if self.gas_closure_name not in self.gas_full_names:
            raise ValueError("gas_closure_name must be in gas_full_names")

        if not set(self.liq_active_names).issubset(set(self.liq_full_names)):
            raise ValueError("liq_active_names must be a subset of liq_full_names")
        if not set(self.gas_active_names).issubset(set(self.gas_full_names)):
            raise ValueError("gas_active_names must be a subset of gas_full_names")

        if len(self.liq_full_to_reduced) != len(self.liq_full_names):
            raise ValueError("liq_full_to_reduced length must match liq_full_names")
        if len(self.gas_full_to_reduced) != len(self.gas_full_names):
            raise ValueError("gas_full_to_reduced length must match gas_full_names")
        if len(self.liq_reduced_to_full) != len(self.liq_active_names):
            raise ValueError("liq_reduced_to_full length must match liq_active_names")
        if len(self.gas_reduced_to_full) != len(self.gas_active_names):
            raise ValueError("gas_reduced_to_full length must match gas_active_names")
        if len(self.liq_full_to_gas_full) != len(self.liq_full_names):
            raise ValueError("liq_full_to_gas_full length must match liq_full_names")
        if np.any(self.liq_full_to_gas_full < 0) or np.any(self.liq_full_to_gas_full >= len(self.gas_full_names)):
            raise ValueError("liq_full_to_gas_full must contain valid gas full indices")

        self._validate_reduced_mapping(
            full_names=self.liq_full_names,
            active_names=self.liq_active_names,
            full_to_reduced=self.liq_full_to_reduced,
            reduced_to_full=self.liq_reduced_to_full,
            mapping_name="liquid",
        )
        self._validate_reduced_mapping(
            full_names=self.gas_full_names,
            active_names=self.gas_active_names,
            full_to_reduced=self.gas_full_to_reduced,
            reduced_to_full=self.gas_reduced_to_full,
            mapping_name="gas",
        )

    @staticmethod
    def _validate_reduced_mapping(
        *,
        full_names: tuple[str, ...],
        active_names: tuple[str, ...],
        full_to_reduced: IntArray,
        reduced_to_full: IntArray,
        mapping_name: str,
    ) -> None:
        if reduced_to_full.size == 0:
            if np.any(full_to_reduced != -1):
                raise ValueError(f"{mapping_name} full_to_reduced must be all -1 when reduced set is empty")
            return

        if np.any(reduced_to_full < 0) or np.any(reduced_to_full >= len(full_names)):
            raise ValueError(f"{mapping_name} reduced_to_full contains out-of-range indices")
        if len(np.unique(reduced_to_full)) != len(reduced_to_full):
            raise ValueError(f"{mapping_name} reduced_to_full must not contain duplicates")

        reduced_names = tuple(full_names[idx] for idx in reduced_to_full.tolist())
        if reduced_names != active_names:
            raise ValueError(
                f"{mapping_name} active_names must match full_names indexed by reduced_to_full"
            )

        valid_full_to_reduced = set(range(len(reduced_to_full))) | {-1}
        if not set(full_to_reduced.tolist()).issubset(valid_full_to_reduced):
            raise ValueError(f"{mapping_name} full_to_reduced contains invalid reduced indices")

        mapped_full_indices = np.where(full_to_reduced >= 0)[0]
        if len(mapped_full_indices) != len(reduced_to_full):
            raise ValueError(f"{mapping_name} full_to_reduced marks extra mapped species")
        if set(mapped_full_indices.tolist()) != set(reduced_to_full.tolist()):
            raise ValueError(
                f"{mapping_name} full_to_reduced / reduced_to_full do not select the same full species"
            )

        for red_idx, full_idx in enumerate(reduced_to_full.tolist()):
            if full_to_reduced[full_idx] != red_idx:
                raise ValueError(
                    f"{mapping_name} full_to_reduced and reduced_to_full are inconsistent at full index {full_idx}"
                )

    @property
    def n_liq_full(self) -> int:
        return len(self.liq_full_names)

    @property
    def n_liq_red(self) -> int:
        return len(self.liq_active_names)

    @property
    def n_gas_full(self) -> int:
        return len(self.gas_full_names)

    @property
    def n_gas_red(self) -> int:
        return len(self.gas_active_names)

    @property
    def is_single_component_liquid(self) -> bool:
        return self.n_liq_full == 1


@dataclass(slots=True, kw_only=True, frozen=True)
class RunConfig:
    """Fully normalized run configuration, without runtime state."""

    case_name: str
    case_description: str | None
    paths: CasePaths
    mesh: MeshConfig
    initialization: InitializationConfig
    species: SpeciesControlConfig
    species_maps: SpeciesMaps
    time_stepper: TimeStepperConfig
    outer_stepper: OuterStepperConfig
    inner_solver: InnerSolverConfig
    recovery: RecoveryConfig
    diagnostics: DiagnosticsConfig
    output: OutputConfig
    validation: ValidationConfig
    unknowns_profile: str
    gas_phase_name: str = "gas"
    liquid_model_name: str = "default"
    equilibrium_model_name: str = "default"

    def __post_init__(self) -> None:
        if not self.case_name:
            raise ValueError("case_name must be non-empty")
        if self.case_description is not None and self.case_description.strip() == "":
            raise ValueError("case_description must be non-empty when provided")
        if not self.gas_phase_name:
            raise ValueError("gas_phase_name must be non-empty")
        if not self.liquid_model_name:
            raise ValueError("liquid_model_name must be non-empty")
        if not self.equilibrium_model_name:
            raise ValueError("equilibrium_model_name must be non-empty")
        if self.unknowns_profile not in {"U_A", "U_B"}:
            raise ValueError("unknowns_profile must be one of {'U_A', 'U_B'}")

    @property
    def pressure(self) -> float:
        return self.initialization.gas_pressure

    @property
    def ambient_temperature(self) -> float:
        return self.initialization.gas_temperature

    @property
    def a0(self) -> float:
        return self.mesh.a0

    @property
    def r_end(self) -> float:
        return self.mesh.r_end


@dataclass(slots=True, kw_only=True, frozen=True)
class RegionSlices:
    """Cell index partitions for liquid / gas-near / gas-far regions."""

    liq: slice
    gas_near: slice
    gas_far: slice
    gas_all: slice

    def __post_init__(self) -> None:
        for name, item in (
            ("liq", self.liq),
            ("gas_near", self.gas_near),
            ("gas_far", self.gas_far),
            ("gas_all", self.gas_all),
        ):
            if item.step not in (None, 1):
                raise ValueError(f"{name} step must be None or 1")
            if _slice_start(item) < 0:
                raise ValueError(f"{name} start must be >= 0")
            if _slice_stop(item) < _slice_start(item):
                raise ValueError(f"{name} stop must be >= start")

        if _slice_start(self.liq) != 0:
            raise ValueError("liq slice must start at 0")
        if _slice_stop(self.liq) != _slice_start(self.gas_all):
            raise ValueError("liq slice must end where gas_all begins")
        if _slice_start(self.gas_near) != _slice_start(self.gas_all):
            raise ValueError("gas_near must begin at gas_all.start")
        if _slice_stop(self.gas_near) != _slice_start(self.gas_far):
            raise ValueError("gas_near and gas_far must be contiguous")
        if _slice_stop(self.gas_far) != _slice_stop(self.gas_all):
            raise ValueError("gas_far must end at gas_all.stop")


@dataclass(slots=True, kw_only=True, frozen=True)
class GeometryState:
    """Outer-level geometry snapshot for a single step / outer iteration."""

    t: float
    dt: float
    a: float
    dot_a: float
    r_end: float
    step_index: int
    outer_iter_index: int

    def __post_init__(self) -> None:
        if not np.isfinite(self.t):
            raise ValueError("t must be finite")
        _check_positive("dt", self.dt)
        _check_positive("a", self.a)
        if not np.isfinite(self.dot_a):
            raise ValueError("dot_a must be finite")
        if not np.isfinite(self.r_end) or self.r_end <= self.a:
            raise ValueError("r_end must be finite and greater than a")
        _check_nonnegative_int("step_index", self.step_index)
        _check_nonnegative_int("outer_iter_index", self.outer_iter_index)

    def is_first_step(self) -> bool:
        return self.step_index == 0


@dataclass(slots=True, kw_only=True, frozen=True)
class Mesh1D:
    """Current-geometry 1D spherical finite-volume mesh."""

    r_faces: FloatArray  # [m], shape (n_faces,)
    r_centers: FloatArray  # [m], shape (n_cells,)
    volumes: FloatArray  # [m^3], shape (n_cells,)
    face_areas: FloatArray  # [m^2], shape (n_faces,)
    dr: FloatArray  # [m], shape (n_cells,)
    region_slices: RegionSlices
    face_owner_phase: IntArray | None = None  # phase id per face, shape (n_faces,)
    interface_face_index: int | None = None
    interface_cell_liq: int | None = None
    interface_cell_gas: int | None = None

    def __post_init__(self) -> None:
        r_faces = _as_float_array("r_faces", self.r_faces, ndim=1)
        r_centers = _as_float_array("r_centers", self.r_centers, ndim=1)
        volumes = _as_float_array("volumes", self.volumes, ndim=1)
        face_areas = _as_float_array("face_areas", self.face_areas, ndim=1)
        dr = _as_float_array("dr", self.dr, ndim=1)
        object.__setattr__(self, "r_faces", r_faces)
        object.__setattr__(self, "r_centers", r_centers)
        object.__setattr__(self, "volumes", volumes)
        object.__setattr__(self, "face_areas", face_areas)
        object.__setattr__(self, "dr", dr)
        _check_all_finite("r_faces", r_faces)
        _check_all_finite("r_centers", r_centers)
        _check_all_finite("volumes", volumes)
        _check_all_finite("face_areas", face_areas)
        _check_all_finite("dr", dr)

        if r_faces.size < 2:
            raise ValueError("r_faces must contain at least two faces")
        _check_strictly_increasing("r_faces", r_faces)
        _check_strictly_increasing("r_centers", r_centers)
        if len(r_centers) != len(volumes) or len(r_centers) != len(dr):
            raise ValueError("r_centers, volumes, and dr must have the same length")
        if len(face_areas) != len(r_faces):
            raise ValueError("face_areas length must match r_faces")
        if len(r_centers) != len(r_faces) - 1:
            raise ValueError("r_centers length must equal len(r_faces) - 1")
        if np.any(volumes <= 0.0):
            raise ValueError("volumes must be strictly positive")
        if np.any(face_areas < 0.0):
            raise ValueError("face_areas must be non-negative")
        if np.any(dr <= 0.0):
            raise ValueError("dr must be strictly positive")
        if np.any(r_centers <= r_faces[:-1]) or np.any(r_centers >= r_faces[1:]):
            raise ValueError("r_centers must lie strictly inside each face interval")

        if self.face_owner_phase is not None:
            face_owner_phase = _as_int_array("face_owner_phase", self.face_owner_phase, ndim=1)
            if len(face_owner_phase) != len(r_faces):
                raise ValueError("face_owner_phase length must match r_faces")
            object.__setattr__(self, "face_owner_phase", face_owner_phase)

        max_cells = len(r_centers)
        for name, idx, upper in (
            ("interface_face_index", self.interface_face_index, len(r_faces)),
            ("interface_cell_liq", self.interface_cell_liq, max_cells),
            ("interface_cell_gas", self.interface_cell_gas, max_cells),
        ):
            if idx is not None and not (0 <= idx < upper):
                raise ValueError(f"{name} must be in [0, {upper})")

        for name, slc in (
            ("liq", self.region_slices.liq),
            ("gas_near", self.region_slices.gas_near),
            ("gas_far", self.region_slices.gas_far),
            ("gas_all", self.region_slices.gas_all),
        ):
            if _slice_stop(slc) > max_cells:
                raise ValueError(f"{name} slice exceeds n_cells")
        if _slice_stop(self.region_slices.gas_all) != max_cells:
            raise ValueError("gas_all slice must end at n_cells")

    @property
    def n_cells(self) -> int:
        return int(self.r_centers.shape[0])

    @property
    def n_faces(self) -> int:
        return int(self.r_faces.shape[0])

    @property
    def liq_slice(self) -> slice:
        return self.region_slices.liq

    @property
    def gas_slice(self) -> slice:
        return self.region_slices.gas_all

    @property
    def n_liq(self) -> int:
        return _slice_stop(self.region_slices.liq) - _slice_start(self.region_slices.liq)

    @property
    def n_gas(self) -> int:
        return _slice_stop(self.region_slices.gas_all) - _slice_start(self.region_slices.gas_all)

    def same_geometry(self, other: "Mesh1D") -> bool:
        return (
            np.array_equal(self.r_faces, other.r_faces)
            and np.array_equal(self.r_centers, other.r_centers)
            and np.array_equal(self.volumes, other.volumes)
            and np.array_equal(self.face_areas, other.face_areas)
            and np.array_equal(self.dr, other.dr)
            and self.region_slices == other.region_slices
            and self.interface_face_index == other.interface_face_index
            and self.interface_cell_liq == other.interface_cell_liq
            and self.interface_cell_gas == other.interface_cell_gas
            and (
                (self.face_owner_phase is None and other.face_owner_phase is None)
                or (
                    self.face_owner_phase is not None
                    and other.face_owner_phase is not None
                    and np.array_equal(self.face_owner_phase, other.face_owner_phase)
                )
            )
        )


@dataclass(slots=True, kw_only=True, frozen=True)
class ControlSurfaceMetrics:
    """Current-geometry control-surface velocity data on faces and cells."""

    v_c_faces: FloatArray  # [m/s], shape (n_faces,)
    v_c_cells: FloatArray  # [m/s], shape (n_cells,)
    interface_face_index: int
    region2_outer_face_index: int

    def __post_init__(self) -> None:
        v_c_faces = _as_float_array("v_c_faces", self.v_c_faces, ndim=1)
        v_c_cells = _as_float_array("v_c_cells", self.v_c_cells, ndim=1)
        object.__setattr__(self, "v_c_faces", v_c_faces)
        object.__setattr__(self, "v_c_cells", v_c_cells)
        _check_all_finite("v_c_faces", v_c_faces)
        _check_all_finite("v_c_cells", v_c_cells)
        if len(v_c_faces) < 2:
            raise ValueError("v_c_faces must contain at least two entries")
        if len(v_c_cells) != len(v_c_faces) - 1:
            raise ValueError("v_c_cells length must equal len(v_c_faces) - 1")
        if not (0 <= self.interface_face_index < len(v_c_faces)):
            raise ValueError("interface_face_index must be in [0, n_faces)")
        if not (0 <= self.region2_outer_face_index < len(v_c_faces)):
            raise ValueError("region2_outer_face_index must be in [0, n_faces)")
        if self.region2_outer_face_index < self.interface_face_index:
            raise ValueError("region2_outer_face_index must be >= interface_face_index")


@dataclass(slots=True, kw_only=True)
class InterfaceState:
    """Full-order interface state shared across interface and bulk coupling."""

    Ts: float  # [K]
    mpp: float  # [kg/m^2/s]
    Ys_g_full: FloatArray  # [-], shape (n_gas_species_full,)
    Ys_l_full: FloatArray  # [-], shape (n_liq_species_full,)

    def __post_init__(self) -> None:
        if not np.isfinite(self.Ts):
            raise ValueError("Ts must be finite")
        if not np.isfinite(self.mpp):
            raise ValueError("mpp must be finite")
        self.Ys_g_full = _as_float_array("Ys_g_full", self.Ys_g_full, ndim=1)
        self.Ys_l_full = _as_float_array("Ys_l_full", self.Ys_l_full, ndim=1)
        _require_nonempty("Ys_g_full", self.Ys_g_full)
        _require_nonempty("Ys_l_full", self.Ys_l_full)
        _check_all_finite("Ys_g_full", self.Ys_g_full)
        _check_all_finite("Ys_l_full", self.Ys_l_full)


@dataclass(slots=True, kw_only=True)
class State:
    """Full-order liquid / gas / interface state snapshot."""

    Tl: FloatArray  # [K], shape (n_liq_cells,)
    Yl_full: FloatArray  # [-], shape (n_liq_cells, n_liq_species_full)
    Tg: FloatArray  # [K], shape (n_gas_cells,)
    Yg_full: FloatArray  # [-], shape (n_gas_cells, n_gas_species_full)
    interface: InterfaceState
    rho_l: FloatArray | None = None  # [kg/m^3], shape (n_liq_cells,)
    rho_g: FloatArray | None = None  # [kg/m^3], shape (n_gas_cells,)
    hl: FloatArray | None = None  # [J/kg], shape (n_liq_cells,)
    hg: FloatArray | None = None  # [J/kg], shape (n_gas_cells,)
    Xg_full: FloatArray | None = None  # [-], shape (n_gas_cells, n_gas_species_full)
    time: float | None = None
    state_id: str | None = None

    def __post_init__(self) -> None:
        self.Tl = _as_float_array("Tl", self.Tl, ndim=1)
        self.Yl_full = _as_float_array("Yl_full", self.Yl_full, ndim=2)
        self.Tg = _as_float_array("Tg", self.Tg, ndim=1)
        self.Yg_full = _as_float_array("Yg_full", self.Yg_full, ndim=2)
        _check_all_finite("Tl", self.Tl)
        _check_all_finite("Yl_full", self.Yl_full)
        _check_all_finite("Tg", self.Tg)
        _check_all_finite("Yg_full", self.Yg_full)
        self.validate_shapes()

        if self.rho_l is not None:
            self.rho_l = _as_float_array("rho_l", self.rho_l, ndim=1)
            _check_all_finite("rho_l", self.rho_l)
            _check_array_row_count("rho_l", self.rho_l, self.n_liq_cells)
        if self.rho_g is not None:
            self.rho_g = _as_float_array("rho_g", self.rho_g, ndim=1)
            _check_all_finite("rho_g", self.rho_g)
            _check_array_row_count("rho_g", self.rho_g, self.n_gas_cells)
        if self.hl is not None:
            self.hl = _as_float_array("hl", self.hl, ndim=1)
            _check_all_finite("hl", self.hl)
            _check_array_row_count("hl", self.hl, self.n_liq_cells)
        if self.hg is not None:
            self.hg = _as_float_array("hg", self.hg, ndim=1)
            _check_all_finite("hg", self.hg)
            _check_array_row_count("hg", self.hg, self.n_gas_cells)
        if self.Xg_full is not None:
            self.Xg_full = _as_float_array("Xg_full", self.Xg_full, ndim=2)
            _check_all_finite("Xg_full", self.Xg_full)
            if self.Xg_full.shape != self.Yg_full.shape:
                raise ValueError("Xg_full shape must match Yg_full shape")
        if self.time is not None and not np.isfinite(self.time):
            raise ValueError("time must be finite when provided")

    @property
    def n_liq_cells(self) -> int:
        return int(self.Tl.shape[0])

    @property
    def n_gas_cells(self) -> int:
        return int(self.Tg.shape[0])

    @property
    def n_liq_species_full(self) -> int:
        return int(self.Yl_full.shape[1])

    @property
    def n_gas_species_full(self) -> int:
        return int(self.Yg_full.shape[1])

    def validate_shapes(self) -> None:
        if self.Yl_full.shape[0] != self.Tl.shape[0]:
            raise ValueError("Yl_full row count must match Tl length")
        if self.Yg_full.shape[0] != self.Tg.shape[0]:
            raise ValueError("Yg_full row count must match Tg length")
        if self.Yl_full.shape[1] != self.interface.Ys_l_full.shape[0]:
            raise ValueError("Yl_full species dimension must match interface.Ys_l_full length")
        if self.Yg_full.shape[1] != self.interface.Ys_g_full.shape[0]:
            raise ValueError("Yg_full species dimension must match interface.Ys_g_full length")

    def copy_shallow(self) -> State:
        """Return a shallow dataclass copy. NumPy array buffers are shared."""
        return replace(self)

    def with_interface(self, interface_new: InterfaceState) -> State:
        return replace(self, interface=interface_new)


@dataclass(slots=True, kw_only=True, frozen=True)
class Props:
    """Bulk cell-centered property results, without model handles."""

    rho_l: FloatArray
    cp_l: FloatArray
    h_l: FloatArray
    k_l: FloatArray
    mu_l: FloatArray
    D_l: FloatArray | None

    rho_g: FloatArray
    cp_g: FloatArray
    h_g: FloatArray
    k_g: FloatArray
    mu_g: FloatArray
    D_g: FloatArray | None

    diagnostics: dict[str, Any]

    def __post_init__(self) -> None:
        rho_l = _as_float_array("rho_l", self.rho_l, ndim=1)
        cp_l = _as_float_array("cp_l", self.cp_l, ndim=1)
        h_l = _as_float_array("h_l", self.h_l, ndim=1)
        k_l = _as_float_array("k_l", self.k_l, ndim=1)
        mu_l = _as_float_array("mu_l", self.mu_l, ndim=1)

        rho_g = _as_float_array("rho_g", self.rho_g, ndim=1)
        cp_g = _as_float_array("cp_g", self.cp_g, ndim=1)
        h_g = _as_float_array("h_g", self.h_g, ndim=1)
        k_g = _as_float_array("k_g", self.k_g, ndim=1)
        mu_g = _as_float_array("mu_g", self.mu_g, ndim=1)

        _check_all_finite("rho_l", rho_l)
        _check_all_finite("cp_l", cp_l)
        _check_all_finite("h_l", h_l)
        _check_all_finite("k_l", k_l)
        _check_all_finite("mu_l", mu_l)
        _check_all_finite("rho_g", rho_g)
        _check_all_finite("cp_g", cp_g)
        _check_all_finite("h_g", h_g)
        _check_all_finite("k_g", k_g)
        _check_all_finite("mu_g", mu_g)

        _check_same_length("rho_l", rho_l, "cp_l", cp_l)
        _check_same_length("rho_l", rho_l, "h_l", h_l)
        _check_same_length("rho_l", rho_l, "k_l", k_l)
        _check_same_length("rho_l", rho_l, "mu_l", mu_l)

        _check_same_length("rho_g", rho_g, "cp_g", cp_g)
        _check_same_length("rho_g", rho_g, "h_g", h_g)
        _check_same_length("rho_g", rho_g, "k_g", k_g)
        _check_same_length("rho_g", rho_g, "mu_g", mu_g)

        if np.any(rho_l <= 0.0):
            raise ValueError("rho_l must be strictly positive")
        if np.any(cp_l <= 0.0):
            raise ValueError("cp_l must be strictly positive")
        if np.any(k_l <= 0.0):
            raise ValueError("k_l must be strictly positive")
        if np.any(mu_l <= 0.0):
            raise ValueError("mu_l must be strictly positive")

        if np.any(rho_g <= 0.0):
            raise ValueError("rho_g must be strictly positive")
        if np.any(cp_g <= 0.0):
            raise ValueError("cp_g must be strictly positive")
        if np.any(k_g <= 0.0):
            raise ValueError("k_g must be strictly positive")
        if np.any(mu_g <= 0.0):
            raise ValueError("mu_g must be strictly positive")

        object.__setattr__(self, "rho_l", rho_l)
        object.__setattr__(self, "cp_l", cp_l)
        object.__setattr__(self, "h_l", h_l)
        object.__setattr__(self, "k_l", k_l)
        object.__setattr__(self, "mu_l", mu_l)
        object.__setattr__(self, "rho_g", rho_g)
        object.__setattr__(self, "cp_g", cp_g)
        object.__setattr__(self, "h_g", h_g)
        object.__setattr__(self, "k_g", k_g)
        object.__setattr__(self, "mu_g", mu_g)

        if self.D_l is not None:
            D_l = _as_float_array("D_l", self.D_l, ndim=2)
            _check_all_finite("D_l", D_l)
            if np.any(D_l <= 0.0):
                raise ValueError("D_l must be strictly positive")
            _check_array_row_count("D_l", D_l, rho_l.shape[0])
            object.__setattr__(self, "D_l", D_l)

        if self.D_g is not None:
            D_g = _as_float_array("D_g", self.D_g, ndim=2)
            _check_all_finite("D_g", D_g)
            if np.any(D_g <= 0.0):
                raise ValueError("D_g must be strictly positive")
            _check_array_row_count("D_g", D_g, rho_g.shape[0])
            object.__setattr__(self, "D_g", D_g)

        if not isinstance(self.diagnostics, dict):
            raise TypeError("diagnostics must be a dict")

    @property
    def hl(self) -> FloatArray:
        return self.h_l

    @property
    def hg(self) -> FloatArray:
        return self.h_g

    @property
    def Dl(self) -> FloatArray | None:
        return self.D_l

    @property
    def Dg(self) -> FloatArray | None:
        return self.D_g

    @property
    def has_liquid_diffusion(self) -> bool:
        return self.D_l is not None

    @property
    def has_gas_diffusion(self) -> bool:
        return self.D_g is not None


@dataclass(slots=True, kw_only=True)
class ConservativeContents:
    """Phase-wise conservative cell contents on the current geometry."""

    mass_l: FloatArray  # [kg], shape (n_liq_cells,)
    species_mass_l: FloatArray  # [kg], shape (n_liq_cells, n_liq_species_full)
    enthalpy_l: FloatArray  # [J], shape (n_liq_cells,)
    mass_g: FloatArray  # [kg], shape (n_gas_cells,)
    species_mass_g: FloatArray  # [kg], shape (n_gas_cells, n_gas_species_full)
    enthalpy_g: FloatArray  # [J], shape (n_gas_cells,)

    def __post_init__(self) -> None:
        self.mass_l = _as_float_array("mass_l", self.mass_l, ndim=1)
        self.species_mass_l = _as_float_array("species_mass_l", self.species_mass_l, ndim=2)
        self.enthalpy_l = _as_float_array("enthalpy_l", self.enthalpy_l, ndim=1)
        self.mass_g = _as_float_array("mass_g", self.mass_g, ndim=1)
        self.species_mass_g = _as_float_array("species_mass_g", self.species_mass_g, ndim=2)
        self.enthalpy_g = _as_float_array("enthalpy_g", self.enthalpy_g, ndim=1)
        _check_all_finite("mass_l", self.mass_l)
        _check_all_finite("species_mass_l", self.species_mass_l)
        _check_all_finite("enthalpy_l", self.enthalpy_l)
        _check_all_finite("mass_g", self.mass_g)
        _check_all_finite("species_mass_g", self.species_mass_g)
        _check_all_finite("enthalpy_g", self.enthalpy_g)

        _check_array_row_count("species_mass_l", self.species_mass_l, self.n_liq_cells)
        _check_array_row_count("enthalpy_l", self.enthalpy_l, self.n_liq_cells)
        _check_array_row_count("species_mass_g", self.species_mass_g, self.n_gas_cells)
        _check_array_row_count("enthalpy_g", self.enthalpy_g, self.n_gas_cells)

    @property
    def n_liq_cells(self) -> int:
        return int(self.mass_l.shape[0])

    @property
    def n_gas_cells(self) -> int:
        return int(self.mass_g.shape[0])


def _check_mesh_state_consistency(*, mesh: "Mesh1D", state: "State", contents: "ConservativeContents") -> None:
    if contents.n_liq_cells != state.n_liq_cells:
        raise ValueError("Liquid cell count mismatch between contents and state")
    if contents.n_gas_cells != state.n_gas_cells:
        raise ValueError("Gas cell count mismatch between contents and state")
    if mesh.n_liq != state.n_liq_cells:
        raise ValueError("mesh.n_liq must match state.n_liq_cells")
    if mesh.n_gas != state.n_gas_cells:
        raise ValueError("mesh.n_gas must match state.n_gas_cells")


@dataclass(slots=True, kw_only=True)
class StateTransferRecord:
    """Current-geometry state-transfer record for one inner-entry contract.

    This is the formal type for the Phase A transition contract.
    ``OldStateOnCurrentGeometry`` is retained as a compatibility alias for
    pre-migration code paths.
    """

    contents: ConservativeContents
    state: State
    geometry: GeometryState
    mesh: Mesh1D
    source_outer_iter_index: int | None = None
    identity_transfer: bool = False

    def __post_init__(self) -> None:
        _check_mesh_state_consistency(mesh=self.mesh, state=self.state, contents=self.contents)
        if self.source_outer_iter_index is not None:
            _check_nonnegative_int("source_outer_iter_index", self.source_outer_iter_index)


OldStateOnCurrentGeometry = StateTransferRecord


@dataclass(slots=True, kw_only=True, init=False)
class OuterIterState:
    """Frozen geometry snapshot for one outer predictor-corrector iteration."""

    geometry: GeometryState
    mesh: Mesh1D
    entry_state: State | None = None
    entry_transfer: StateTransferRecord | None = None
    entry_source: str | None = None
    predicted_from_accepted: bool

    def __init__(
        self,
        *,
        geometry: GeometryState,
        mesh: Mesh1D,
        entry_state: State | None = None,
        entry_transfer: StateTransferRecord | None = None,
        entry_source: str | None = None,
        predicted_from_accepted: bool,
        old_state_current_geom: StateTransferRecord | None = None,
    ) -> None:
        if entry_transfer is not None and old_state_current_geom is not None and entry_transfer is not old_state_current_geom:
            raise ValueError("entry_transfer and old_state_current_geom must refer to the same transfer record")
        self.geometry = geometry
        self.mesh = mesh
        self.entry_state = entry_state
        self.entry_transfer = entry_transfer if entry_transfer is not None else old_state_current_geom
        self.entry_source = entry_source
        self.predicted_from_accepted = predicted_from_accepted
        self.__post_init__()

    @staticmethod
    def _normalize_entry_source(
        *,
        entry_source: str | None,
        entry_transfer: StateTransferRecord | None,
    ) -> str:
        if entry_source is None:
            return "transfer_from_previous_outer" if entry_transfer is not None else "accepted_time_level"
        if entry_source not in ("accepted_time_level", "transfer_from_previous_outer"):
            raise ValueError("entry_source must be 'accepted_time_level' or 'transfer_from_previous_outer'")
        return entry_source

    def __post_init__(self) -> None:
        self.entry_source = self._normalize_entry_source(
            entry_source=self.entry_source,
            entry_transfer=self.entry_transfer,
        )
        if self.entry_transfer is not None and not self.entry_transfer.mesh.same_geometry(self.mesh):
            raise ValueError("entry_transfer.mesh must match current outer mesh exactly")
        if self.entry_source == "accepted_time_level":
            if self.entry_transfer is not None:
                raise ValueError("accepted_time_level entry must not carry entry_transfer")
        else:
            if self.entry_transfer is None:
                raise ValueError("transfer_from_previous_outer entry must provide entry_transfer")
        if self.entry_state is not None:
            if self.mesh.n_liq != self.entry_state.n_liq_cells:
                raise ValueError("entry_state liquid cell count must match mesh.n_liq")
            if self.mesh.n_gas != self.entry_state.n_gas_cells:
                raise ValueError("entry_state gas cell count must match mesh.n_gas")

    @property
    def old_state_current_geom(self) -> StateTransferRecord | None:
        """Read-only compatibility alias for ``entry_transfer``."""

        return self.entry_transfer


@dataclass(slots=True, kw_only=True)
class StepContext:
    """Step-level runtime snapshot, separate from configuration."""

    accepted_state: State
    accepted_geometry: GeometryState
    accepted_mesh: Mesh1D
    dt_try: float
    step_index: int
    retry_count_for_current_state: int
    last_failure_class: str | None = None
    accepted_state_id: str | None = None

    def __post_init__(self) -> None:
        _check_positive("dt_try", self.dt_try)
        _check_nonnegative_int("step_index", self.step_index)
        _check_nonnegative_int("retry_count_for_current_state", self.retry_count_for_current_state)
        if self.accepted_mesh.n_liq != self.accepted_state.n_liq_cells:
            raise ValueError("accepted_mesh.n_liq must match accepted_state.n_liq_cells")
        if self.accepted_mesh.n_gas != self.accepted_state.n_gas_cells:
            raise ValueError("accepted_mesh.n_gas must match accepted_state.n_gas_cells")

    @property
    def accepted_state_n(self) -> State:
        return self.accepted_state

    @property
    def accepted_geometry_n(self) -> GeometryState:
        return self.accepted_geometry

    @property
    def accepted_mesh_n(self) -> Mesh1D:
        return self.accepted_mesh


__all__ = [
    "CasePaths",
    "ConservativeContents",
    "ControlSurfaceMetrics",
    "DiagnosticsConfig",
    "FieldSplitBulkConfig",
    "FieldSplitConfig",
    "FieldSplitIfaceConfig",
    "FloatArray",
    "GeometryState",
    "InitializationConfig",
    "InnerSolverConfig",
    "IntArray",
    "InterfaceState",
    "Mesh1D",
    "MeshConfig",
    "OldStateOnCurrentGeometry",
    "OuterIterState",
    "OuterStepperConfig",
    "OutputConfig",
    "PathLike",
    "Props",
    "RecoveryConfig",
    "RegionSlices",
    "RunConfig",
    "SpeciesControlConfig",
    "SpeciesMaps",
    "StateTransferRecord",
    "State",
    "StepContext",
    "TimeStepperConfig",
    "ValidationConfig",
]
