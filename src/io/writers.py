from __future__ import annotations

"""Run-output writers built on top of the canonical OutputLayout contract."""

from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence
import csv
import importlib.util
import json
import math
import sys

import numpy as np
import yaml


def _load_output_layout_symbols():
    module_path = Path(__file__).with_name("output_layout.py")
    spec = importlib.util.spec_from_file_location("paper_v1_io_output_layout", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("failed to load io/output_layout.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.OutputLayout, module.FailureLayout, module.step_snapshot_path


OutputLayout, FailureLayout, step_snapshot_path = _load_output_layout_symbols()


def _load_runtime_bridge_module():
    module_path = Path(__file__).resolve().parents[1] / "driver" / "runtime_evp.py"
    if not module_path.exists():
        return None
    module_name = "paper_v1_driver_runtime_evp"
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


STEP_DIAG_COLUMNS = [
    "step",
    "t_old",
    "t_new",
    "dt",
    "linear_converged",
    "linear_method",
    "linear_n_iter",
    "linear_residual_norm",
    "linear_rel_residual",
    "nonlinear_converged",
    "nonlinear_method",
    "nonlinear_n_iter",
    "nonlinear_residual_norm",
    "nonlinear_residual_inf",
    "outer_iter_count",
    "retries_used",
    "outer_converged",
    "eps_dot_a",
    "dot_a_phys",
    "snes_reason",
    "ksp_reason",
    "failure_class",
    "reject_reason",
    "Ts",
    "Rd",
    "mpp",
    "Tg_min",
    "Tg_max",
    "energy_balance_if",
    "mass_balance_rd",
]

SCALARS_COLUMNS = [
    "step",
    "t",
    "dt",
    "Ts",
    "Rd",
    "mpp",
    "m_dot",
    "Tg_min",
    "Tg_max",
    "linear_converged",
    "nonlinear_converged",
    "nonlinear_n_iter",
    "nonlinear_residual_inf",
    "outer_iter_count",
    "outer_converged",
    "retries_used",
    "eps_dot_a",
    "dot_a_phys",
    "snes_reason",
    "ksp_reason",
    "energy_balance_if",
    "mass_balance_rd",
]

INTERFACE_DIAG_COLUMNS = [
    "step",
    "Ts",
    "Rd",
    "mpp",
    "m_dot",
    "psat",
    "sum_y_cond",
    "sum_y_cond_raw",
    "psat_over_P",
    "psat_source",
    "hvap_source",
    "latent_source",
    "q_g",
    "q_l",
    "q_diff",
    "latent",
    "energy_res",
    "energy_sign",
    "guard_active",
    "clamp_active",
    "penalty_used",
    "penalty_reason",
    "n_penalty_residual",
    "ls_lambda",
    "ls_clip_ts",
    "ls_clip_mpp",
    "ls_shrink_count",
    "regime",
    "regime_locked",
    "Ts_upper",
    "Ts_guard",
    "Ts_hard",
]


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _normalize_scalar(value: Any) -> str | float | int | bool:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
        return value
    if value is None:
        return ""
    return str(value)


def _json_sanitize(obj: Any) -> Any:
    if is_dataclass(obj):
        return _json_sanitize(asdict(obj))
    if isinstance(obj, Mapping):
        return {str(k): _json_sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_sanitize(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return _json_sanitize(obj.item())
    if isinstance(obj, float):
        if math.isnan(obj):
            return "nan"
        if math.isinf(obj):
            return "inf" if obj > 0 else "-inf"
    if hasattr(obj, "__dict__") and not isinstance(obj, (str, bytes)):
        return {str(k): _json_sanitize(v) for k, v in vars(obj).items() if not str(k).startswith("_")}
    return obj


def _failure_class_value(value: Any) -> Any:
    if value is None:
        return ""
    return getattr(value, "value", value)


def _snapshot_scalar(value: Any, *, default: float = np.nan) -> float:
    if isinstance(value, np.generic):
        value = value.item()
    if value is None:
        return float(default)
    if isinstance(value, str) and value == "":
        return float(default)
    try:
        out = float(value)
    except Exception:
        return float(default)
    if not math.isfinite(out):
        return float(default)
    return out


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    _ensure_parent(path)
    path.write_text(
        json.dumps(_json_sanitize(payload), indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _append_csv_row(path: Path, row: Mapping[str, Any], *, field_order: Sequence[str]) -> None:
    _ensure_parent(path)
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(field_order), extrasaction="ignore")
        if not exists:
            writer.writeheader()
        normalized = {field: _normalize_scalar(row.get(field, "")) for field in field_order}
        writer.writerow(normalized)


def _is_root_writer(runtime: object | None = None, is_root: bool | None = None) -> bool:
    if is_root is not None:
        return bool(is_root)
    if runtime is not None:
        if getattr(runtime, "is_root", None) is not None:
            return bool(runtime.is_root)
        if getattr(runtime, "rank", None) is not None:
            return int(runtime.rank) == 0
    return True


def _diag_extra(diag: object) -> dict[str, Any]:
    extra = getattr(diag, "extra", None)
    return dict(extra) if isinstance(extra, Mapping) else {}


def _diag_value(diag: object, name: str, default: Any = "") -> Any:
    if hasattr(diag, name):
        return getattr(diag, name)
    return _diag_extra(diag).get(name, default)


def _infer_step_id(diag: object) -> int:
    for key in ("step", "step_id", "step_index"):
        value = _diag_value(diag, key, None)
        if value is not None and value != "":
            return int(value)
    raise ValueError("step identifier is required in diag.step, diag.step_id, or diag.step_index")


def _infer_m_dot(diag: object) -> Any:
    interface_diag = _diag_extra(diag).get("interface_diag", {})
    if isinstance(interface_diag, Mapping) and interface_diag.get("m_dot") is not None:
        return interface_diag["m_dot"]
    rd = _diag_value(diag, "Rd", None)
    mpp = _diag_value(diag, "mpp", None)
    if rd in (None, "") or mpp in (None, ""):
        return ""
    rd_f = float(rd)
    mpp_f = float(mpp)
    if not math.isfinite(rd_f) or not math.isfinite(mpp_f):
        return ""
    return 4.0 * math.pi * rd_f * rd_f * mpp_f


def _diag_to_step_row(step_id: int, diag: object) -> dict[str, Any]:
    failure_class = _failure_class_value(_diag_value(diag, "failure_class", ""))
    return {
        "step": int(step_id),
        "t_old": _diag_value(diag, "t_old", ""),
        "t_new": _diag_value(diag, "t_new", ""),
        "dt": _diag_value(diag, "dt", ""),
        "linear_converged": _diag_value(diag, "linear_converged", ""),
        "linear_method": _diag_value(diag, "linear_method", ""),
        "linear_n_iter": _diag_value(diag, "linear_n_iter", ""),
        "linear_residual_norm": _diag_value(diag, "linear_residual_norm", ""),
        "linear_rel_residual": _diag_value(diag, "linear_rel_residual", ""),
        "nonlinear_converged": _diag_value(diag, "nonlinear_converged", ""),
        "nonlinear_method": _diag_value(diag, "nonlinear_method", ""),
        "nonlinear_n_iter": _diag_value(diag, "nonlinear_n_iter", ""),
        "nonlinear_residual_norm": _diag_value(diag, "nonlinear_residual_norm", ""),
        "nonlinear_residual_inf": _diag_value(diag, "nonlinear_residual_inf", ""),
        "outer_iter_count": _diag_value(diag, "outer_iter_count", _diag_value(diag, "outer_iterations", "")),
        "retries_used": _diag_value(diag, "retries_used", ""),
        "outer_converged": _diag_value(diag, "outer_converged", ""),
        "eps_dot_a": _diag_value(diag, "eps_dot_a", ""),
        "dot_a_phys": _diag_value(diag, "dot_a_phys", ""),
        "snes_reason": _diag_value(diag, "snes_reason", ""),
        "ksp_reason": _diag_value(diag, "ksp_reason", ""),
        "failure_class": failure_class,
        "reject_reason": _diag_value(diag, "reject_reason", ""),
        "Ts": _diag_value(diag, "Ts", ""),
        "Rd": _diag_value(diag, "Rd", ""),
        "mpp": _diag_value(diag, "mpp", ""),
        "Tg_min": _diag_value(diag, "Tg_min", ""),
        "Tg_max": _diag_value(diag, "Tg_max", ""),
        "energy_balance_if": _diag_value(diag, "energy_balance_if", ""),
        "mass_balance_rd": _diag_value(diag, "mass_balance_rd", ""),
    }


def _diag_to_scalars_row(step_id: int, t: float, state: object, diag: object, cfg: object | None = None) -> dict[str, Any]:
    _ = (state, cfg)
    return {
        "step": int(step_id),
        "t": float(t),
        "dt": _diag_value(diag, "dt", ""),
        "Ts": _diag_value(diag, "Ts", ""),
        "Rd": _diag_value(diag, "Rd", ""),
        "mpp": _diag_value(diag, "mpp", ""),
        "m_dot": _infer_m_dot(diag),
        "Tg_min": _diag_value(diag, "Tg_min", ""),
        "Tg_max": _diag_value(diag, "Tg_max", ""),
        "linear_converged": _diag_value(diag, "linear_converged", ""),
        "nonlinear_converged": _diag_value(diag, "nonlinear_converged", ""),
        "nonlinear_n_iter": _diag_value(diag, "nonlinear_n_iter", ""),
        "nonlinear_residual_inf": _diag_value(diag, "nonlinear_residual_inf", ""),
        "outer_iter_count": _diag_value(diag, "outer_iter_count", _diag_value(diag, "outer_iterations", "")),
        "outer_converged": _diag_value(diag, "outer_converged", ""),
        "retries_used": _diag_value(diag, "retries_used", ""),
        "eps_dot_a": _diag_value(diag, "eps_dot_a", ""),
        "dot_a_phys": _diag_value(diag, "dot_a_phys", ""),
        "snes_reason": _diag_value(diag, "snes_reason", ""),
        "ksp_reason": _diag_value(diag, "ksp_reason", ""),
        "energy_balance_if": _diag_value(diag, "energy_balance_if", ""),
        "mass_balance_rd": _diag_value(diag, "mass_balance_rd", ""),
    }


def _diag_to_interface_row(step_id: int, diag: object) -> dict[str, Any]:
    interface_diag = _diag_extra(diag).get("interface_diag", {})
    payload = dict(interface_diag) if isinstance(interface_diag, Mapping) else {}
    payload["step"] = int(step_id)
    return payload


def _config_to_plain_data(cfg: object) -> Any:
    if is_dataclass(cfg):
        return asdict(cfg)
    if isinstance(cfg, Mapping):
        return {str(k): _config_to_plain_data(v) for k, v in cfg.items()}
    if isinstance(cfg, (list, tuple)):
        return [_config_to_plain_data(v) for v in cfg]
    if hasattr(cfg, "__dict__") and not isinstance(cfg, (str, bytes)):
        return {str(k): _config_to_plain_data(v) for k, v in vars(cfg).items() if not str(k).startswith("_")}
    return cfg


def _require_output_layout_from_cfg(cfg: object):
    if getattr(cfg, "_output_layout", None) is not None:
        return cfg._output_layout
    if getattr(cfg, "output_layout", None) is not None:
        return cfg.output_layout
    runtime = getattr(cfg, "runtime", None)
    if runtime is not None and getattr(runtime, "output_layout", None) is not None:
        return runtime.output_layout
    paths = getattr(cfg, "paths", None)
    if paths is not None and getattr(paths, "_output_layout", None) is not None:
        return paths._output_layout
    runtime_bridge = _load_runtime_bridge_module()
    if runtime_bridge is not None:
        get_bound_output_layout = getattr(runtime_bridge, "get_bound_output_layout", None)
        if callable(get_bound_output_layout):
            bound = get_bound_output_layout(cfg)
            if bound is not None:
                return bound
    raise ValueError("cfg must carry an injected OutputLayout via _output_layout, output_layout, runtime.output_layout, or paths._output_layout")


def _runtime_from_cfg(cfg: object) -> object | None:
    runtime = getattr(cfg, "runtime", None)
    if runtime is not None:
        return runtime
    runtime_bridge = _load_runtime_bridge_module()
    if runtime_bridge is None:
        return None
    get_bound_runtime = getattr(runtime_bridge, "get_bound_runtime", None)
    if callable(get_bound_runtime):
        return get_bound_runtime(cfg)
    return None


def _state_scalar(state: object, name: str, *, interface_name: str | None = None, default: Any = "") -> Any:
    if hasattr(state, name):
        return getattr(state, name)
    if interface_name is not None:
        interface = getattr(state, "interface", None)
        if interface is not None and hasattr(interface, interface_name):
            return getattr(interface, interface_name)
    return default


def _state_array(state: object, *names: str) -> np.ndarray:
    for name in names:
        value = getattr(state, name, None)
        if value is not None:
            return np.asarray(value, dtype=np.float64)
    raise ValueError(f"state must expose one of: {', '.join(names)}")


def _grid_faces(grid: object) -> np.ndarray:
    value = getattr(grid, "r_f", None)
    if value is None:
        value = getattr(grid, "r_faces", None)
    if value is None:
        raise ValueError("grid must expose r_f or r_faces")
    return np.asarray(value, dtype=np.float64)


def _grid_centers(grid: object) -> np.ndarray:
    value = getattr(grid, "r_c", None)
    if value is None:
        value = getattr(grid, "r_centers", None)
    if value is None:
        raise ValueError("grid must expose r_c or r_centers")
    return np.asarray(value, dtype=np.float64)


def _grid_meta(grid: object, layout: object | None = None) -> tuple[Any, Any, Any]:
    n_liq = getattr(grid, "Nl", None)
    n_gas = getattr(grid, "Ng", None)
    iface_f = getattr(grid, "iface_f", None)
    if n_liq is not None and n_gas is not None and iface_f is not None:
        return int(n_liq), int(n_gas), int(iface_f)
    region_slices = getattr(grid, "region_slices", None)
    if region_slices is not None:
        liq_slice = getattr(region_slices, "liq", None)
        gas_all = getattr(region_slices, "gas_all", None)
        if liq_slice is not None and gas_all is not None:
            liq_stop = int(liq_slice.stop)
            gas_len = int(gas_all.stop) - int(gas_all.start)
            iface = getattr(grid, "interface_face_index", liq_stop)
            return liq_stop, gas_len, int(iface)
    if layout is not None:
        n_liq_cells = getattr(layout, "n_liq_cells", None)
        n_gas_cells = getattr(layout, "n_gas_cells", None)
        if n_liq_cells is not None and n_gas_cells is not None:
            return int(n_liq_cells), int(n_gas_cells), ""
    return "", "", ""


def write_config_echo(cfg, output_layout: OutputLayout, *, is_root: bool = True) -> None:
    if not _is_root_writer(is_root=is_root):
        return
    _ensure_parent(output_layout.config_echo_yaml)
    payload = _json_sanitize(_config_to_plain_data(cfg))
    output_layout.config_echo_yaml.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def write_metadata_echo(runtime, output_layout: OutputLayout, *, is_root: bool = True) -> None:
    if not _is_root_writer(runtime=runtime, is_root=is_root):
        return
    payload = {
        "case_id": output_layout.case_id,
        "run_id": output_layout.run_id,
        "backend": getattr(runtime, "backend", ""),
        "petsc_enabled": getattr(runtime, "petsc_enabled", ""),
        "mpi_size": getattr(runtime, "mpi_size", getattr(runtime, "size", "")),
        "mpi_rank_root": 0,
        "output_root": output_layout.root,
        "created_at": getattr(runtime, "created_at", ""),
        "schema_version": getattr(runtime, "schema_version", ""),
        "parallel_mode": getattr(runtime, "parallel_mode", ""),
        "single_rank_only_boundary_active": getattr(runtime, "single_rank_only_boundary_active", ""),
    }
    _write_json(output_layout.metadata_json, payload)


def write_run_summary(output_layout: OutputLayout, *, summary: Mapping[str, Any], is_root: bool = True) -> None:
    if not _is_root_writer(is_root=is_root):
        return
    _write_json(output_layout.run_summary_json, dict(summary))


def append_step_diag_row(output_layout: OutputLayout, *, step_id: int, diag, is_root: bool = True) -> None:
    if not _is_root_writer(is_root=is_root):
        return
    _append_csv_row(output_layout.step_diag_csv, _diag_to_step_row(step_id, diag), field_order=STEP_DIAG_COLUMNS)


def append_scalars_row(
    output_layout: OutputLayout,
    *,
    step_id: int,
    t: float,
    state,
    diag,
    cfg=None,
    is_root: bool = True,
) -> None:
    if not _is_root_writer(is_root=is_root):
        return
    _append_csv_row(
        output_layout.scalars_csv,
        _diag_to_scalars_row(step_id, t, state, diag, cfg=cfg),
        field_order=SCALARS_COLUMNS,
    )


def append_interface_diag_row(output_layout: OutputLayout, *, step_id: int, diag, is_root: bool = True) -> None:
    if not _is_root_writer(is_root=is_root):
        return
    _append_csv_row(
        output_layout.interface_diag_csv,
        _diag_to_interface_row(step_id, diag),
        field_order=INTERFACE_DIAG_COLUMNS,
    )


def write_mapping_json(output_layout: OutputLayout, *, grid, layout, cfg, is_root: bool = True) -> None:
    if not _is_root_writer(is_root=is_root):
        return
    if output_layout.mapping_json.exists():
        return
    species_maps = getattr(cfg, "species_maps", None)
    species_cfg = getattr(cfg, "species", None)
    n_liq, n_gas, iface_f = _grid_meta(grid, layout)
    payload = {
        "case_id": output_layout.case_id,
        "run_id": output_layout.run_id,
        "layout_version": "phase7_v1",
        "unknown_layout_size": getattr(layout, "total_size", ""),
        "species_order": {
            "gas_full": tuple(getattr(species_maps, "gas_full_names", ()) or ()),
            "gas_reduced": tuple(getattr(species_maps, "gas_active_names", ()) or ()),
            "liquid": tuple(getattr(species_maps, "liq_full_names", ()) or ()),
        },
        "closure_species": {
            "gas": getattr(species_cfg, "gas_closure_species", getattr(species_maps, "gas_closure_name", "")),
            "liquid": getattr(species_cfg, "liquid_closure_species", getattr(species_maps, "liq_closure_name", "")),
        },
        "grid": {
            "n_liq": n_liq,
            "n_gas": n_gas,
            "iface_f": iface_f,
            "segment_policy": getattr(grid, "segment_policy", ""),
        },
        "fields": {
            "Tl": {"shape": "n_liq", "location": "center", "units": "K"},
            "Tg": {"shape": "n_gas", "location": "center", "units": "K"},
            "Yl": {"shape": "n_liq x n_liq_species", "location": "center", "units": "1"},
            "Yg": {"shape": "n_gas x n_gas_species", "location": "center", "units": "1"},
            "Ts": {"shape": "scalar", "location": "interface", "units": "K"},
            "Rd": {"shape": "scalar", "location": "interface", "units": "m"},
            "mpp": {"shape": "scalar", "location": "interface", "units": "kg m^-2 s^-1"},
            "r_c": {"shape": "n_total", "location": "center", "units": "m"},
            "r_f": {"shape": "n_faces", "location": "face", "units": "m"},
            "Nl": {"shape": "scalar", "location": "metadata", "units": "count"},
            "Ng": {"shape": "scalar", "location": "metadata", "units": "count"},
            "iface_f": {"shape": "scalar", "location": "metadata", "units": "index"},
            "rho_l": {"shape": "n_liq", "location": "center", "units": "kg m^-3", "optional": True},
            "rho_g": {"shape": "n_gas", "location": "center", "units": "kg m^-3", "optional": True},
            "h_l": {"shape": "n_liq", "location": "center", "units": "J kg^-1", "optional": True},
            "h_g": {"shape": "n_gas", "location": "center", "units": "J kg^-1", "optional": True},
            "layout_total_size": {"shape": "scalar", "location": "metadata", "units": "count", "optional": True},
        },
        "conventions": {
            "radial_normal": "outward",
            "heat_flux_sign": "positive_to_gas",
            "mpp_sign": "positive_evaporation",
        },
        "snapshot_filename_pattern": "step_{step_id:06d}_time_{t:.6e}.npz",
    }
    _write_json(output_layout.mapping_json, payload)


def write_step_snapshot(
    output_layout: OutputLayout,
    *,
    step_id: int,
    t: float,
    grid,
    state,
    props=None,
    layout=None,
    diag=None,
    is_root: bool = True,
) -> Path | None:
    if not _is_root_writer(is_root=is_root):
        return None
    out = step_snapshot_path(output_layout, step_id=step_id, t=t)
    _ensure_parent(out)
    r_c = _grid_centers(grid)
    r_f = _grid_faces(grid)
    n_liq, n_gas, iface_f = _grid_meta(grid, layout)
    payload: dict[str, Any] = {
        "step_id": int(step_id),
        "t": float(t),
        "Ts": _snapshot_scalar(_state_scalar(state, "Ts", interface_name="Ts")),
        "Rd": _snapshot_scalar(_state_scalar(state, "Rd", interface_name="Rd", default=_diag_value(diag, "Rd", np.nan))),
        "mpp": _snapshot_scalar(_state_scalar(state, "mpp", interface_name="mpp")),
        "Tl": np.asarray(getattr(state, "Tl"), dtype=np.float64),
        "Tg": np.asarray(getattr(state, "Tg"), dtype=np.float64),
        "Yl": _state_array(state, "Yl", "Yl_full"),
        "Yg": _state_array(state, "Yg", "Yg_full"),
        "r_c": r_c,
        "r_f": r_f,
        "Nl": n_liq,
        "Ng": n_gas,
        "iface_f": iface_f,
    }
    if props is not None:
        for src_name, out_name in (("rho_l", "rho_l"), ("rho_g", "rho_g"), ("h_l", "h_l"), ("h_g", "h_g")):
            value = getattr(props, src_name, None)
            if value is not None:
                payload[out_name] = np.asarray(value, dtype=np.float64)
    if layout is not None and getattr(layout, "total_size", None) is not None:
        payload["layout_total_size"] = int(layout.total_size)
    np.savez_compressed(out, **payload)
    return out


def write_step_scalars(cfg, t, state, diag) -> None:
    output_layout = _require_output_layout_from_cfg(cfg)
    is_root = _is_root_writer(runtime=_runtime_from_cfg(cfg))
    append_scalars_row(
        output_layout,
        step_id=_infer_step_id(diag),
        t=float(t),
        state=state,
        diag=diag,
        cfg=cfg,
        is_root=is_root,
    )


def write_interface_diag(cfg, diag) -> None:
    output_layout = _require_output_layout_from_cfg(cfg)
    is_root = _is_root_writer(runtime=_runtime_from_cfg(cfg))
    append_interface_diag_row(
        output_layout,
        step_id=_infer_step_id(diag),
        diag=diag,
        is_root=is_root,
    )


__all__ = [
    "INTERFACE_DIAG_COLUMNS",
    "SCALARS_COLUMNS",
    "STEP_DIAG_COLUMNS",
    "append_interface_diag_row",
    "append_scalars_row",
    "append_step_diag_row",
    "write_config_echo",
    "write_interface_diag",
    "write_mapping_json",
    "write_metadata_echo",
    "write_run_summary",
    "write_step_scalars",
    "write_step_snapshot",
]
