from __future__ import annotations

"""Normalize a schema-valid raw config into lightweight program data objects.

This module resolves runtime paths, derives species bookkeeping, expands
full-order initialization vectors, and constructs ``RunConfig``. It does not
create mesh objects, build unknown layouts, read full property databases, or
store heavy mechanism handles in the normalized config.
"""

from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .types import (
    CasePaths,
    DiagnosticsConfig,
    FieldSplitBulkConfig,
    FieldSplitConfig,
    FieldSplitIfaceConfig,
    InitializationConfig,
    InnerSolverConfig,
    MeshConfig,
    OutputConfig,
    RecoveryConfig,
    RunConfig,
    SpeciesControlConfig,
    SpeciesMaps,
    TimeStepperConfig,
    ValidationConfig,
    OuterStepperConfig,
)


DEFAULT_GAS_PHASE_NAME = "gas"
DEFAULT_LIQUID_MODEL_NAME = "default"
DEFAULT_INTERFACE_EQUILIBRIUM_MODEL = "default"


class PreprocessError(ValueError):
    """Raised when a schema-valid raw config cannot be normalized into RunConfig."""


def _resolve_case_root(source_path: str | Path) -> Path:
    cfg_path = Path(source_path).expanduser().resolve()
    if not cfg_path.exists():
        raise PreprocessError(f"Config source path does not exist: {cfg_path}")
    if not cfg_path.is_file():
        raise PreprocessError(f"Config source path is not a file: {cfg_path}")
    return cfg_path.parent


def _resolve_path_from_case_root(path_like: str | Path, *, case_root: Path) -> Path:
    path = Path(path_like).expanduser()
    if not path.is_absolute():
        path = (case_root / path).resolve()
    else:
        path = path.resolve()
    return path


def _resolve_external_paths(raw_cfg: Mapping[str, Any], *, case_root: Path, source_path: str | Path) -> CasePaths:
    paths_cfg = raw_cfg["paths"]
    config_path = Path(source_path).expanduser().resolve()
    liquid_database_path = _resolve_path_from_case_root(
        paths_cfg["liquid_database_path"],
        case_root=case_root,
    )
    gas_mechanism_path = _resolve_path_from_case_root(
        paths_cfg["gas_mechanism_path"],
        case_root=case_root,
    )
    output_root = _resolve_path_from_case_root(
        paths_cfg["output_root"],
        case_root=case_root,
    )

    for label, path in (
        ("liquid_database_path", liquid_database_path),
        ("gas_mechanism_path", gas_mechanism_path),
    ):
        if not path.exists():
            raise PreprocessError(f"{label} does not exist: {path}")
        if not path.is_file():
            raise PreprocessError(f"{label} is not a file: {path}")

    return CasePaths(
        config_path=config_path,
        case_root=case_root,
        gas_mechanism_path=gas_mechanism_path,
        liquid_database_path=liquid_database_path,
        output_root=output_root,
    )


def _load_gas_species_names(
    mechanism_path: Path,
    *,
    phase_name: str = DEFAULT_GAS_PHASE_NAME,
) -> tuple[str, ...]:
    try:
        import cantera as ct
    except ImportError as exc:
        raise PreprocessError("Cantera is required to load gas mechanism species names") from exc

    try:
        gas = ct.Solution(str(mechanism_path), phase_name)
    except Exception as exc:  # pragma: no cover - exercised with monkeypatch in unit tests
        raise PreprocessError(
            f"Failed to load gas mechanism '{mechanism_path}' with phase '{phase_name}': {exc}"
        ) from exc

    species_names = tuple(gas.species_names)
    if not species_names:
        raise PreprocessError(f"Gas mechanism '{mechanism_path}' returned an empty species list")
    return species_names


def _derive_liquid_species_names(raw_cfg: Mapping[str, Any]) -> tuple[str, ...]:
    liquid_composition = raw_cfg["initialization"]["liquid_composition"]
    liquid_full_names = tuple(liquid_composition.keys())
    if len(liquid_full_names) == 0:
        raise PreprocessError("initialization.liquid_composition must contain at least one species")
    return liquid_full_names


def _validate_species_control_against_sources(
    *,
    gas_full_names: tuple[str, ...],
    liquid_full_names: tuple[str, ...],
    species_cfg: Mapping[str, Any],
    initialization_cfg: Mapping[str, Any],
) -> None:
    gas_closure_species = species_cfg["gas_closure_species"]
    liquid_closure_species = species_cfg.get("liquid_closure_species")
    liquid_to_gas_name_map = species_cfg["liquid_to_gas_species_map"]
    gas_composition = initialization_cfg["gas_composition"]
    y_vap_if0 = initialization_cfg["Y_vap_if0"]

    gas_full_set = set(gas_full_names)
    liquid_full_set = set(liquid_full_names)

    if gas_closure_species not in gas_full_set:
        raise PreprocessError("gas_closure_species must be present in gas mechanism species")

    unknown_gas_initial = set(gas_composition.keys()) - gas_full_set
    if unknown_gas_initial:
        raise PreprocessError(
            f"initialization.gas_composition contains unknown gas species: {sorted(unknown_gas_initial)}"
        )

    if set(liquid_to_gas_name_map.keys()) != liquid_full_set:
        raise PreprocessError(
            "species.liquid_to_gas_species_map keys must match liquid full species exactly"
        )

    mapped_targets = tuple(liquid_to_gas_name_map[name] for name in liquid_full_names)
    unknown_targets = set(mapped_targets) - gas_full_set
    if unknown_targets:
        raise PreprocessError(
            f"species.liquid_to_gas_species_map contains gas targets not present in mechanism: {sorted(unknown_targets)}"
        )
    if len(set(mapped_targets)) != len(mapped_targets):
        raise PreprocessError("species.liquid_to_gas_species_map must be one-to-one")

    if len(liquid_full_names) == 1:
        only_liquid = liquid_full_names[0]
        if liquid_closure_species is not None and liquid_closure_species != only_liquid:
            raise PreprocessError(
                "liquid_closure_species must match the single liquid species when provided"
            )
    else:
        if not isinstance(liquid_closure_species, str) or not liquid_closure_species:
            raise PreprocessError("multicomponent liquid requires liquid_closure_species")
        if liquid_closure_species not in liquid_full_set:
            raise PreprocessError("liquid_closure_species must be one of the liquid full species")

    mapped_vapor_set = set(mapped_targets)
    invalid_y_vap_if0 = set(y_vap_if0.keys()) - mapped_vapor_set
    if invalid_y_vap_if0:
        raise PreprocessError(
            f"initialization.Y_vap_if0 contains species outside mapped vapor set: {sorted(invalid_y_vap_if0)}"
        )


def _build_reduced_mapping_arrays(
    full_names: tuple[str, ...],
    closure_name: str | None,
) -> tuple[tuple[str, ...], np.ndarray, np.ndarray]:
    if closure_name is None:
        active_names: tuple[str, ...] = ()
        full_to_reduced = np.full(len(full_names), -1, dtype=np.int64)
        reduced_to_full = np.array([], dtype=np.int64)
        return active_names, full_to_reduced, reduced_to_full

    active_names_list: list[str] = []
    full_to_reduced = np.full(len(full_names), -1, dtype=np.int64)
    reduced_to_full_list: list[int] = []

    for full_idx, name in enumerate(full_names):
        if name == closure_name:
            continue
        red_idx = len(active_names_list)
        active_names_list.append(name)
        full_to_reduced[full_idx] = red_idx
        reduced_to_full_list.append(full_idx)

    return (
        tuple(active_names_list),
        full_to_reduced,
        np.asarray(reduced_to_full_list, dtype=np.int64),
    )


def _build_liquid_to_gas_full_index_map(
    liquid_full_names: tuple[str, ...],
    gas_full_names: tuple[str, ...],
    liquid_to_gas_name_map: Mapping[str, str],
) -> np.ndarray:
    gas_name_to_index = {name: idx for idx, name in enumerate(gas_full_names)}
    return np.asarray(
        [gas_name_to_index[liquid_to_gas_name_map[liq_name]] for liq_name in liquid_full_names],
        dtype=np.int64,
    )


def _build_species_maps(
    *,
    gas_full_names: tuple[str, ...],
    liquid_full_names: tuple[str, ...],
    gas_closure_name: str,
    liquid_closure_name: str | None,
    liquid_to_gas_name_map: Mapping[str, str],
) -> SpeciesMaps:
    gas_active_names, gas_full_to_reduced, gas_reduced_to_full = _build_reduced_mapping_arrays(
        gas_full_names,
        gas_closure_name,
    )
    liq_active_names, liq_full_to_reduced, liq_reduced_to_full = _build_reduced_mapping_arrays(
        liquid_full_names,
        liquid_closure_name,
    )
    liq_full_to_gas_full = _build_liquid_to_gas_full_index_map(
        liquid_full_names,
        gas_full_names,
        liquid_to_gas_name_map,
    )
    return SpeciesMaps(
        liq_full_names=liquid_full_names,
        liq_active_names=liq_active_names,
        liq_closure_name=liquid_closure_name,
        gas_full_names=gas_full_names,
        gas_active_names=gas_active_names,
        gas_closure_name=gas_closure_name,
        liq_full_to_reduced=liq_full_to_reduced,
        liq_reduced_to_full=liq_reduced_to_full,
        gas_full_to_reduced=gas_full_to_reduced,
        gas_reduced_to_full=gas_reduced_to_full,
        liq_full_to_gas_full=liq_full_to_gas_full,
    )


def _build_recovery_config(raw_recovery: Mapping[str, Any]) -> RecoveryConfig:
    """Build RecoveryConfig from a schema-validated raw recovery dict."""
    return RecoveryConfig(
        rho_min=float(raw_recovery["rho_min"]),
        m_min=float(raw_recovery["m_min"]),
        species_recovery_eps_abs=float(raw_recovery["species_recovery_eps_abs"]),
        Y_sum_tol=float(raw_recovery["Y_sum_tol"]),
        Y_hard_tol=float(raw_recovery["Y_hard_tol"]),
        h_abs_tol=float(raw_recovery["h_abs_tol"]),
        h_rel_tol=float(raw_recovery["h_rel_tol"]),
        h_check_tol=float(raw_recovery["h_check_tol"]),
        T_step_tol=float(raw_recovery["T_step_tol"]),
        T_min_l=float(raw_recovery["T_min_l"]),
        T_max_l=float(raw_recovery["T_max_l"]),
        T_min_g=float(raw_recovery["T_min_g"]),
        T_max_g=float(raw_recovery["T_max_g"]),
        liquid_h_inv_max_iter=int(raw_recovery["liquid_h_inv_max_iter"]),
        cp_min=float(raw_recovery["cp_min"]),
        gas_h_inv_max_iter=int(raw_recovery["gas_h_inv_max_iter"]),
        use_cantera_hpy_first=bool(raw_recovery["use_cantera_hpy_first"]),
    )


def _validate_mass_fraction_vector(
    *,
    name: str,
    vec: np.ndarray,
    sum_mode: str,
    sum_tol: float,
    neg_tol: float,
) -> None:
    """Validate a full-order mass fraction vector.

    Parameters
    ----------
    name:
        Human-readable label used in error messages.
    vec:
        Full-order mass fraction array.
    sum_mode:
        ``"eq_one"`` requires ``|sum(vec) - 1| <= sum_tol``.
        ``"leq_one"`` requires ``0 <= sum(vec) <= 1 + sum_tol``.
    sum_tol:
        Absolute tolerance on the sum check.
    neg_tol:
        Absolute tolerance below which negative values are rejected
        (i.e. values < -neg_tol are an error).
    """
    if not np.all(np.isfinite(vec)):
        raise PreprocessError(
            f"Mass fraction vector '{name}' contains non-finite values"
        )
    if np.any(vec < -neg_tol):
        raise PreprocessError(
            f"Mass fraction vector '{name}' contains values below -{neg_tol:.3e}: "
            f"min={float(np.min(vec)):.3e}"
        )
    total = float(np.sum(vec))
    if sum_mode == "eq_one":
        if abs(total - 1.0) > sum_tol:
            raise PreprocessError(
                f"Mass fraction vector '{name}' must sum to 1.0 within tolerance "
                f"{sum_tol:.3e}: sum={total:.10f}"
            )
    elif sum_mode == "leq_one":
        if total < -sum_tol or total > 1.0 + sum_tol:
            raise PreprocessError(
                f"Mass fraction vector '{name}' must satisfy 0 <= sum <= 1 within "
                f"tolerance {sum_tol:.3e}: sum={total:.10f}"
            )
    else:
        raise ValueError(f"Unknown sum_mode: {sum_mode!r}")


def _build_full_mass_fraction_vector(
    *,
    full_names: tuple[str, ...],
    provided_mass_fractions: Mapping[str, float],
) -> np.ndarray:
    full_name_to_index = {name: idx for idx, name in enumerate(full_names)}
    unknown_species = set(provided_mass_fractions.keys()) - set(full_names)
    if unknown_species:
        raise PreprocessError(f"Unknown species in mass-fraction input: {sorted(unknown_species)}")

    vector = np.zeros(len(full_names), dtype=np.float64)
    for name, value in provided_mass_fractions.items():
        vector[full_name_to_index[name]] = float(value)
    return vector


def _build_initialization_config(
    *,
    raw_init: Mapping[str, Any],
    gas_full_names: tuple[str, ...],
    liquid_full_names: tuple[str, ...],
    mapped_vapor_gas_names: tuple[str, ...],
    y_sum_tol: float,
    species_neg_tol: float,
) -> InitializationConfig:
    y_vap_if0_keys = set(raw_init["Y_vap_if0"].keys())
    if not y_vap_if0_keys.issubset(set(mapped_vapor_gas_names)):
        raise PreprocessError("initialization.Y_vap_if0 must be a subset of mapped vapor gas species")

    gas_y_full_0 = _build_full_mass_fraction_vector(
        full_names=gas_full_names,
        provided_mass_fractions=raw_init["gas_composition"],
    )
    liquid_y_full_0 = _build_full_mass_fraction_vector(
        full_names=liquid_full_names,
        provided_mass_fractions=raw_init["liquid_composition"],
    )
    y_vap_if0_gas_full = _build_full_mass_fraction_vector(
        full_names=gas_full_names,
        provided_mass_fractions=raw_init["Y_vap_if0"],
    )

    _validate_mass_fraction_vector(
        name="gas_y_full_0",
        vec=gas_y_full_0,
        sum_mode="eq_one",
        sum_tol=y_sum_tol,
        neg_tol=species_neg_tol,
    )
    _validate_mass_fraction_vector(
        name="liquid_y_full_0",
        vec=liquid_y_full_0,
        sum_mode="eq_one",
        sum_tol=y_sum_tol,
        neg_tol=species_neg_tol,
    )
    _validate_mass_fraction_vector(
        name="y_vap_if0_gas_full",
        vec=y_vap_if0_gas_full,
        sum_mode="leq_one",
        sum_tol=y_sum_tol,
        neg_tol=species_neg_tol,
    )

    return InitializationConfig(
        gas_temperature=float(raw_init["gas_temperature"]),
        gas_pressure=float(raw_init["gas_pressure"]),
        liquid_temperature=float(raw_init["liquid_temperature"]),
        gas_y_full_0=gas_y_full_0,
        liquid_y_full_0=liquid_y_full_0,
        y_vap_if0_gas_full=y_vap_if0_gas_full,
        t_init_T=float(raw_init["t_init_T"]),
    )


def _derive_unknowns_profile(*, liquid_full_names: tuple[str, ...]) -> str:
    if len(liquid_full_names) == 1:
        return "U_A"
    return "U_B"


def _normalize_liquid_closure_name(
    liquid_full_names: tuple[str, ...],
    liquid_closure_name: str | None,
) -> str | None:
    if len(liquid_full_names) == 1:
        return None
    return liquid_closure_name


def _normalize_time_stepper_cfg(raw_ts: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "t0": float(raw_ts["t0"]),
        "t_end": float(raw_ts["t_end"]),
        "dt_start": float(raw_ts["dt_start"]),
        "dt_min": float(raw_ts["dt_min"]),
        "dt_max": float(raw_ts["dt_max"]),
        "max_retries_per_step": int(raw_ts["max_retries_per_step"]),
        "accept_growth_factor": float(raw_ts["accept_growth_factor"]),
        "reject_shrink_factor": float(raw_ts["reject_shrink_factor"]),
    }


def _normalize_outer_stepper_cfg(raw_outer: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "outer_max_iter": int(raw_outer["outer_max_iter"]),
        "predictor_mode": str(raw_outer["predictor_mode"]),
        "corrector_mode": str(raw_outer["corrector_mode"]),
        "omega_a": float(raw_outer["omega_a"]),
        "omega_v": float(raw_outer["omega_v"]),
        "outer_convergence_mode": str(raw_outer["outer_convergence_mode"]),
        "outer_convergence_tol": float(raw_outer["outer_convergence_tol"]),
        "eps_ref_dot_a": float(raw_outer["eps_ref_dot_a"]),
    }


def _normalize_inner_solver_cfg(raw_solver: Mapping[str, Any]) -> dict[str, Any]:
    raw_fieldsplit = raw_solver["fieldsplit"]
    raw_bulk = raw_fieldsplit["bulk"]
    raw_iface = raw_fieldsplit["iface"]
    return {
        "snes_type": str(raw_solver["snes_type"]),
        "linesearch_type": str(raw_solver["linesearch_type"]),
        "snes_rtol": float(raw_solver["snes_rtol"]),
        "snes_atol": float(raw_solver["snes_atol"]),
        "snes_stol": float(raw_solver["snes_stol"]),
        "snes_max_it": int(raw_solver["snes_max_it"]),
        "options_prefix": str(raw_solver.get("options_prefix", "")),
        "lag_jacobian": int(raw_solver["lag_jacobian"]),
        "lag_preconditioner": int(raw_solver["lag_preconditioner"]),
        "ksp_type": str(raw_solver["ksp_type"]),
        "pc_type": str(raw_solver["pc_type"]),
        "ksp_rtol": float(raw_solver["ksp_rtol"]),
        "ksp_atol": float(raw_solver["ksp_atol"]),
        "ksp_max_it": int(raw_solver["ksp_max_it"]),
        "restart": int(raw_solver["restart"]),
        "gmres_modified_gram_schmidt": bool(raw_solver["gmres_modified_gram_schmidt"]),
        "gmres_preallocate": bool(raw_solver["gmres_preallocate"]),
        "fieldsplit": FieldSplitConfig(
            scheme=str(raw_fieldsplit["scheme"]),
            type=str(raw_fieldsplit["type"]),
            schur_fact_type=str(raw_fieldsplit["schur_fact_type"]),
            schur_precondition=str(raw_fieldsplit["schur_precondition"]),
            bulk=FieldSplitBulkConfig(
                ksp_type=str(raw_bulk["ksp_type"]),
                pc_type=str(raw_bulk["pc_type"]),
                sub_ksp_type=str(raw_bulk["sub_ksp_type"]),
                sub_pc_type=str(raw_bulk["sub_pc_type"]),
                asm_overlap=int(raw_bulk["asm_overlap"]),
            ),
            iface=FieldSplitIfaceConfig(
                ksp_type=str(raw_iface["ksp_type"]),
                pc_type=str(raw_iface["pc_type"]),
            ),
        ),
    }


def _build_run_config(
    *,
    raw_cfg: Mapping[str, Any],
    paths: CasePaths,
    species_maps: SpeciesMaps,
    initialization: InitializationConfig,
    unknowns_profile: str,
    recovery_cfg: RecoveryConfig,
) -> RunConfig:
    mesh_cfg = raw_cfg["mesh"]
    species_cfg = raw_cfg["species"]

    liquid_full_names = species_maps.liq_full_names
    liquid_closure_name = _normalize_liquid_closure_name(
        liquid_full_names,
        species_cfg.get("liquid_closure_species"),
    )
    species_control = SpeciesControlConfig(
        gas_closure_species=species_cfg["gas_closure_species"],
        liquid_closure_species=liquid_closure_name,
        liquid_to_gas_species_map=dict(species_cfg["liquid_to_gas_species_map"]),
    )

    time_stepper_cfg = _normalize_time_stepper_cfg(raw_cfg["time_stepper"])
    outer_stepper_cfg = _normalize_outer_stepper_cfg(raw_cfg["outer_stepper"])
    inner_solver_cfg = _normalize_inner_solver_cfg(raw_cfg["solver_inner_petsc"])

    return RunConfig(
        case_name=raw_cfg["case"]["name"],
        case_description=raw_cfg["case"].get("description"),
        paths=paths,
        mesh=MeshConfig(
            a0=float(mesh_cfg["a0"]),
            r_end=float(mesh_cfg["r_end"]),
            n_liq=int(mesh_cfg["n_liq"]),
            n_gas_near=int(mesh_cfg["n_gas_near"]),
            far_stretch_ratio=float(mesh_cfg["far_stretch_ratio"]),
        ),
        initialization=initialization,
        species=species_control,
        species_maps=species_maps,
        time_stepper=TimeStepperConfig(**time_stepper_cfg),
        outer_stepper=OuterStepperConfig(**outer_stepper_cfg),
        inner_solver=InnerSolverConfig(**inner_solver_cfg),
        recovery=recovery_cfg,
        diagnostics=DiagnosticsConfig(**raw_cfg["diagnostics"]),
        output=OutputConfig(**raw_cfg["output"]),
        validation=ValidationConfig(**raw_cfg["validation"]),
        unknowns_profile=unknowns_profile,
        gas_phase_name=DEFAULT_GAS_PHASE_NAME,
        liquid_model_name=DEFAULT_LIQUID_MODEL_NAME,
        equilibrium_model_name=DEFAULT_INTERFACE_EQUILIBRIUM_MODEL,
    )


def normalize_config(
    raw_cfg: Mapping[str, Any],
    *,
    source_path: str | Path,
) -> RunConfig:
    # Build recovery config first: its tolerances are used by _build_initialization_config.
    recovery_cfg = _build_recovery_config(raw_cfg["recovery"])

    case_root = _resolve_case_root(source_path)
    paths = _resolve_external_paths(raw_cfg, case_root=case_root, source_path=source_path)
    gas_full_names = _load_gas_species_names(paths.gas_mechanism_path, phase_name=DEFAULT_GAS_PHASE_NAME)
    liquid_full_names = _derive_liquid_species_names(raw_cfg)
    _validate_species_control_against_sources(
        gas_full_names=gas_full_names,
        liquid_full_names=liquid_full_names,
        species_cfg=raw_cfg["species"],
        initialization_cfg=raw_cfg["initialization"],
    )

    liquid_closure_name = _normalize_liquid_closure_name(
        liquid_full_names,
        raw_cfg["species"].get("liquid_closure_species"),
    )
    species_maps = _build_species_maps(
        gas_full_names=gas_full_names,
        liquid_full_names=liquid_full_names,
        gas_closure_name=raw_cfg["species"]["gas_closure_species"],
        liquid_closure_name=liquid_closure_name,
        liquid_to_gas_name_map=raw_cfg["species"]["liquid_to_gas_species_map"],
    )
    initialization = _build_initialization_config(
        raw_init=raw_cfg["initialization"],
        gas_full_names=gas_full_names,
        liquid_full_names=liquid_full_names,
        mapped_vapor_gas_names=tuple(raw_cfg["species"]["liquid_to_gas_species_map"].values()),
        y_sum_tol=recovery_cfg.Y_sum_tol,
        species_neg_tol=recovery_cfg.species_recovery_eps_abs,
    )
    unknowns_profile = _derive_unknowns_profile(liquid_full_names=liquid_full_names)
    return _build_run_config(
        raw_cfg=raw_cfg,
        paths=paths,
        species_maps=species_maps,
        initialization=initialization,
        unknowns_profile=unknowns_profile,
        recovery_cfg=recovery_cfg,
    )


def _echo_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, tuple):
        return [_echo_value(item) for item in value]
    if isinstance(value, list):
        return [_echo_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _echo_value(item) for key, item in value.items()}
    if is_dataclass(value):
        return {key: _echo_value(item) for key, item in asdict(value).items()}
    return value


def build_normalized_config_echo(run_cfg: RunConfig) -> dict[str, Any]:
    return _echo_value(run_cfg)


__all__ = [
    "DEFAULT_GAS_PHASE_NAME",
    "DEFAULT_INTERFACE_EQUILIBRIUM_MODEL",
    "DEFAULT_LIQUID_MODEL_NAME",
    "PreprocessError",
    "build_normalized_config_echo",
    "normalize_config",
]
