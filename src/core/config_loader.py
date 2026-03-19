from __future__ import annotations

"""Load raw YAML config and perform schema validation only.

This module does not normalize configuration, resolve runtime paths, derive
species maps, or construct ``RunConfig``.
"""

from pathlib import Path
from typing import Any, Mapping

import yaml

from .config_schema import ConfigSchemaError, ConfigValidationError, validate_config_schema


class ConfigLoadError(ValueError):
    """Raised when the raw YAML config cannot be loaded from disk."""


class UniqueKeySafeLoader(yaml.SafeLoader):
    """YAML loader that rejects duplicate mapping keys."""


def _construct_mapping_no_duplicates(
    loader: yaml.SafeLoader,
    node: yaml.nodes.MappingNode,
    deep: bool = False,
) -> dict[str, Any]:
    loader.flatten_mapping(node)
    mapping: dict[str, Any] = {}

    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            key_hash = hash(key)
        except TypeError as exc:
            raise yaml.constructor.ConstructorError(
                context="while constructing a mapping",
                context_mark=node.start_mark,
                problem=f"found unhashable key of type {type(key).__name__}",
                problem_mark=key_node.start_mark,
            ) from exc
        if key_hash is None:
            raise AssertionError("unreachable")
        if key in mapping:
            line = key_node.start_mark.line + 1
            column = key_node.start_mark.column + 1
            raise yaml.constructor.ConstructorError(
                context="while constructing a mapping",
                context_mark=node.start_mark,
                problem=f"Duplicate key '{key}' detected in YAML config at line {line}, column {column}",
                problem_mark=key_node.start_mark,
            )
        mapping[key] = loader.construct_object(value_node, deep=deep)

    return mapping


UniqueKeySafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_mapping_no_duplicates,
)


def _coerce_config_path(path: str | Path) -> Path:
    try:
        return Path(path).expanduser()
    except TypeError as exc:
        raise ConfigLoadError(f"Invalid config path object: {path!r}") from exc


def _validate_config_file_path(path: Path) -> None:
    if not path.exists():
        raise ConfigLoadError(f"Config file does not exist: {path}")
    if not path.is_file():
        raise ConfigLoadError(f"Config path is not a file: {path}")


def _load_yaml_from_path(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            loaded = yaml.load(handle, Loader=UniqueKeySafeLoader)
    except yaml.YAMLError as exc:
        raise ConfigLoadError(f"Failed to parse YAML config '{path}': {exc}") from exc
    except OSError as exc:
        raise ConfigLoadError(f"Failed to read config file '{path}': {exc}") from exc

    if loaded is None:
        raise ConfigLoadError(f"Config file is empty or contains no YAML document: {path}")
    if not isinstance(loaded, Mapping):
        raise ConfigLoadError(f"Top-level YAML document must be a mapping: {path}")

    return dict(loaded)


def validate_loaded_config(raw_cfg: Mapping[str, Any], *, source_path: Path | None = None) -> None:
    try:
        validate_config_schema(raw_cfg)
    except ConfigSchemaError as exc:
        if source_path is None:
            raise
        raise ConfigValidationError(f"Invalid config schema in '{source_path}': {exc}") from exc


def load_raw_config(path: str | Path) -> dict[str, Any]:
    cfg_path = _coerce_config_path(path)
    _validate_config_file_path(cfg_path)
    return _load_yaml_from_path(cfg_path)


def load_and_validate_config(path: str | Path) -> dict[str, Any]:
    cfg_path = _coerce_config_path(path)
    raw_cfg = load_raw_config(cfg_path)
    validate_loaded_config(raw_cfg, source_path=cfg_path)
    return raw_cfg


__all__ = [
    "ConfigLoadError",
    "load_raw_config",
    "load_and_validate_config",
]
