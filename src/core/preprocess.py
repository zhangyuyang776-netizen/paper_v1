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
) -> InitializationConfig:
    y_vap_if0_keys = set(raw_init["Y_vap_if0"].keys())
    if not y_vap_if0_keys.issubset(set(mapped_vapor_gas_names)):
        raise PreprocessError("initialization.Y_vap_if0 must be a subset of mapped vapor gas species")

    return InitializationConfig(
        gas_temperature=float(raw_init["gas_temperature"]),
        gas_pressure=float(raw_init["gas_pressure"]),
        liquid_temperature=float(raw_init["liquid_temperature"]),
        gas_y_full_0=_build_full_mass_fraction_vector(
            full_names=gas_full_names,
            provided_mass_fractions=raw_init["gas_composition"],
        ),
        liquid_y_full_0=_build_full_mass_fraction_vector(
            full_names=liquid_full_names,
            provided_mass_fractions=raw_init["liquid_composition"],
        ),
        y_vap_if0_gas_full=_build_full_mass_fraction_vector(
            full_names=gas_full_names,
            provided_mass_fractions=raw_init["Y_vap_if0"],
        ),
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


def _build_run_config(
    *,
    raw_cfg: Mapping[str, Any],
    paths: CasePaths,
    species_maps: SpeciesMaps,
    initialization: InitializationConfig,
    unknowns_profile: str,
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
        time_stepper=TimeStepperConfig(**raw_cfg["time_stepper"]),
        outer_stepper=OuterStepperConfig(**raw_cfg["outer_stepper"]),
        inner_solver=InnerSolverConfig(**raw_cfg["solver_inner_petsc"]),
        recovery=RecoveryConfig(**raw_cfg["recovery"]),
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
    )
    unknowns_profile = _derive_unknowns_profile(liquid_full_names=liquid_full_names)
    return _build_run_config(
        raw_cfg=raw_cfg,
        paths=paths,
        species_maps=species_maps,
        initialization=initialization,
        unknowns_profile=unknowns_profile,
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
