from __future__ import annotations

"""Failure-attempt artifact writers built on the canonical FailureLayout contract."""

from pathlib import Path
from typing import Any, Mapping
import csv
import importlib.util
import json
import math
import shutil
import sys

import numpy as np


def _load_output_layout_symbols():
    module_path = Path(__file__).with_name("output_layout.py")
    spec = importlib.util.spec_from_file_location("paper_v1_io_output_layout", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("failed to load io/output_layout.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return (
        module.FailureLayout,
        module.OutputLayout,
        module.build_failure_layout,
        module.ensure_failure_dir,
        module.failure_rank_meta_path,
        module.failure_rank_snes_last_x_csv_path,
    )


(
    FailureLayout,
    OutputLayout,
    build_failure_layout,
    ensure_failure_dir,
    failure_rank_meta_path,
    failure_rank_snes_last_x_csv_path,
) = _load_output_layout_symbols()


def _load_runtime_logging_symbols():
    module_path = Path(__file__).with_name("runtime_logging.py")
    spec = importlib.util.spec_from_file_location("paper_v1_io_runtime_logging", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("failed to load io/runtime_logging.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.copy_log_tail, module.copy_jsonl_tail


copy_log_tail, copy_jsonl_tail = _load_runtime_logging_symbols()


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _json_sanitize(obj: Any) -> Any:
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
    if hasattr(obj, "value") and not isinstance(obj, (str, bytes)):
        return _json_sanitize(getattr(obj, "value"))
    if hasattr(obj, "__dict__") and not isinstance(obj, (str, bytes)):
        return {str(k): _json_sanitize(v) for k, v in vars(obj).items() if not str(k).startswith("_")}
    return obj


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    _ensure_parent(path)
    path.write_text(
        json.dumps(_json_sanitize(payload), indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _is_root_writer(runtime: object | None = None, is_root: bool | None = None) -> bool:
    if is_root is not None:
        return bool(is_root)
    if runtime is not None:
        if getattr(runtime, "is_root", None) is not None:
            return bool(runtime.is_root)
        if getattr(runtime, "rank", None) is not None:
            return int(runtime.rank) == 0
    return True


def _cfg_flag(cfg: object | None, section_name: str, field_name: str, default: bool = True) -> bool:
    if cfg is None:
        return default
    section = getattr(cfg, section_name, None)
    if section is None:
        return default
    value = getattr(section, field_name, None)
    if value is None:
        return default
    return bool(value)


def _cfg_int(cfg: object | None, section_name: str, field_name: str, default: int) -> int:
    if cfg is None:
        return int(default)
    section = getattr(cfg, section_name, None)
    if section is None:
        return int(default)
    value = getattr(section, field_name, None)
    if value is None:
        return int(default)
    try:
        return int(value)
    except Exception:
        return int(default)


def _diag_extra(diag: object | None) -> dict[str, Any]:
    extra = getattr(diag, "extra", None) if diag is not None else None
    return dict(extra) if isinstance(extra, Mapping) else {}


def _diag_value(diag: object | None, name: str, default: Any = "") -> Any:
    if diag is not None and hasattr(diag, name):
        return getattr(diag, name)
    return _diag_extra(diag).get(name, default)


def _state_scalar(state: object | None, name: str, default: Any = "") -> Any:
    if state is None:
        return default
    if hasattr(state, name):
        return getattr(state, name)
    interface = getattr(state, "interface", None)
    if interface is not None and hasattr(interface, name):
        return getattr(interface, name)
    return default


def _state_array(state: object | None, primary_name: str, fallback_name: str | None = None) -> np.ndarray | None:
    if state is None:
        return None
    if hasattr(state, primary_name):
        return np.asarray(getattr(state, primary_name), dtype=np.float64)
    if fallback_name is not None and hasattr(state, fallback_name):
        return np.asarray(getattr(state, fallback_name), dtype=np.float64)
    return None


def _grid_faces(grid: object | None) -> np.ndarray | None:
    if grid is None:
        return None
    if hasattr(grid, "r_f"):
        return np.asarray(getattr(grid, "r_f"), dtype=np.float64)
    if hasattr(grid, "r_faces"):
        return np.asarray(getattr(grid, "r_faces"), dtype=np.float64)
    return None


def _grid_centers(grid: object | None) -> np.ndarray | None:
    if grid is None:
        return None
    if hasattr(grid, "r_c"):
        return np.asarray(getattr(grid, "r_c"), dtype=np.float64)
    if hasattr(grid, "r_centers"):
        return np.asarray(getattr(grid, "r_centers"), dtype=np.float64)
    return None


def _grid_meta(grid: object | None) -> dict[str, int | str]:
    if grid is None:
        return {}
    n_liq = getattr(grid, "Nl", None)
    n_gas = getattr(grid, "Ng", None)
    iface_f = getattr(grid, "iface_f", None)
    region_slices = getattr(grid, "region_slices", None)
    if n_liq is None and region_slices is not None and getattr(region_slices, "liq", None) is not None:
        liq_slice = region_slices.liq
        n_liq = int(liq_slice.stop) - int(liq_slice.start)
    if n_gas is None and region_slices is not None and getattr(region_slices, "gas_all", None) is not None:
        gas_slice = region_slices.gas_all
        n_gas = int(gas_slice.stop) - int(gas_slice.start)
    if iface_f is None and getattr(grid, "interface_face_index", None) is not None:
        iface_f = int(getattr(grid, "interface_face_index"))
    return {
        "n_liq": "" if n_liq is None else int(n_liq),
        "n_gas": "" if n_gas is None else int(n_gas),
        "iface_f": "" if iface_f is None else int(iface_f),
    }


def _copy_if_exists(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    _ensure_parent(dst)
    shutil.copyfile(src, dst)
    return True


def _infer_radius(state: object | None, grid: object | None, default: Any = "") -> Any:
    value = _state_scalar(state, "Rd", None)
    if value not in (None, ""):
        return value
    faces = _grid_faces(grid)
    meta = _grid_meta(grid)
    iface_f = meta.get("iface_f")
    if faces is not None and iface_f not in ("", None):
        idx = int(iface_f)
        if 0 <= idx < faces.shape[0]:
            return float(faces[idx])
    return default


def _species_order_payload(cfg: object | None) -> dict[str, Any]:
    species_maps = getattr(cfg, "species_maps", None) if cfg is not None else None
    species = getattr(cfg, "species", None) if cfg is not None else None
    return {
        "gas_full": tuple(getattr(species_maps, "gas_full_names", ()) or ()),
        "gas_reduced": tuple(getattr(species_maps, "gas_active_names", ()) or ()),
        "liquid": tuple(getattr(species_maps, "liq_full_names", ()) or ()),
        "closure_species": {
            "gas": getattr(species_maps, "gas_closure_name", getattr(species, "gas_closure_species", "")),
            "liquid": getattr(species_maps, "liq_closure_name", getattr(species, "liquid_closure_species", "")),
        },
    }


def _build_failure_report_payload(
    *,
    step_id: int,
    retry_id: int,
    t_old: float,
    dt_try: float,
    failure_class: str,
    reject_reason: str,
    diag: object | None = None,
    ctx_meta: Mapping[str, Any] | None = None,
    state_attempt_final: object | None = None,
    log_tail_meta: Mapping[str, Any] | None = None,
    output_layout: OutputLayout | None = None,
    flayout: FailureLayout | None = None,
) -> dict[str, Any]:
    extra = _diag_extra(diag)
    ctx = dict(ctx_meta) if isinstance(ctx_meta, Mapping) else {}
    log_meta = dict(log_tail_meta) if isinstance(log_tail_meta, Mapping) else {}
    return {
        "step_id": int(step_id),
        "retry_id": int(retry_id),
        "t_old": float(t_old),
        "dt_try": float(dt_try),
        "failure_class": str(getattr(failure_class, "value", failure_class)),
        "reject_reason": str(reject_reason),
        "message": ctx.get("message", extra.get("message", _diag_value(diag, "message", ""))),
        "inner_converged": _diag_value(diag, "nonlinear_converged", extra.get("inner_converged", "")),
        "outer_converged": extra.get("outer_converged", ""),
        "snes_reason": extra.get("snes_reason", ctx.get("snes_reason", "")),
        "ksp_reason": extra.get("ksp_reason", ctx.get("ksp_reason", "")),
        "R_total_inf": _diag_value(diag, "nonlinear_residual_inf", extra.get("R_total_inf", "")),
        "R_if_inf": extra.get("R_if_inf", ""),
        "eps_dot_a": extra.get("eps_dot_a", ""),
        "Ts": _state_scalar(state_attempt_final, "Ts", _diag_value(diag, "Ts", "")),
        "mpp": _state_scalar(state_attempt_final, "mpp", _diag_value(diag, "mpp", "")),
        "a_guess": extra.get("a_guess", ""),
        "dot_a_guess": extra.get("dot_a_guess", ""),
        "dot_a_phys": extra.get("dot_a_phys", ""),
        "remap_fail_flag": extra.get("remap_fail_flag", ""),
        "recovery_fail_flag": extra.get("recovery_fail_flag", ""),
        "property_fail_flag": extra.get("property_fail_flag", ""),
        "nonphysical_flag": extra.get("nonphysical_flag", ""),
        "nonlinear_method": _diag_value(diag, "nonlinear_method", ""),
        "nonlinear_n_iter": _diag_value(diag, "nonlinear_n_iter", ""),
        "nonlinear_residual_norm": _diag_value(diag, "nonlinear_residual_norm", ""),
        "nonlinear_residual_inf": _diag_value(diag, "nonlinear_residual_inf", ""),
        "linear_method": _diag_value(diag, "linear_method", ""),
        "linear_n_iter": _diag_value(diag, "linear_n_iter", ""),
        "linear_residual_norm": _diag_value(diag, "linear_residual_norm", ""),
        "penalty_last_reason": ctx.get("penalty_last_reason", ""),
        "penalty_last_stage": ctx.get("penalty_last_stage", ""),
        "ls_postcheck_last": ctx.get("ls_postcheck_last", ""),
        "snes_last_global": ctx.get("snes_last_global", ""),
        "ctx_meta_summary": {k: v for k, v in ctx.items() if k not in {"snes_last_x_owned", "x_owned", "rank_failure_meta"}},
        "has_run_log_tail": bool(log_meta.get("has_run_log_tail", False)),
        "has_run_events_tail": bool(log_meta.get("has_run_events_tail", False)),
        "log_artifacts": {
            "run_log": str(output_layout.run_log) if output_layout is not None else str(log_meta.get("run_log", "")),
            "run_events_jsonl": str(output_layout.run_events_jsonl) if output_layout is not None else str(log_meta.get("run_events_jsonl", "")),
            "run_log_tail": str(flayout.run_log_tail_txt) if flayout is not None else str(log_meta.get("run_log_tail", "")),
            "run_events_tail_jsonl": str(flayout.run_events_tail_jsonl) if flayout is not None else str(log_meta.get("run_events_tail_jsonl", "")),
            "has_run_log": bool(log_meta.get("has_run_log", False)),
            "has_run_events_jsonl": bool(log_meta.get("has_run_events_jsonl", False)),
            "has_run_log_tail": bool(log_meta.get("has_run_log_tail", False)),
            "has_run_events_tail": bool(log_meta.get("has_run_events_tail", False)),
            "run_log_tail_error": log_meta.get("run_log_tail_error", ""),
            "run_events_tail_error": log_meta.get("run_events_tail_error", ""),
        },
    }


def _write_failure_state_snapshot(
    out_path: Path,
    *,
    label: str,
    step_id: int,
    retry_id: int,
    t: float,
    dt_try: float,
    state: object | None = None,
    u_raw: object | None = None,
    grid: object | None = None,
    props: object | None = None,
    layout: object | None = None,
) -> None:
    _ensure_parent(out_path)
    payload: dict[str, Any] = {
        "step_id": int(step_id),
        "retry_id": int(retry_id),
        "t": float(t),
        "dt_try": float(dt_try),
        "label": str(label),
    }

    if state is not None:
        payload["Ts"] = float(_state_scalar(state, "Ts", np.nan))
        payload["Rd"] = float(_infer_radius(state, grid, np.nan))
        payload["mpp"] = float(_state_scalar(state, "mpp", np.nan))
        tl = _state_array(state, "Tl")
        tg = _state_array(state, "Tg")
        yl = _state_array(state, "Yl", fallback_name="Yl_full")
        yg = _state_array(state, "Yg", fallback_name="Yg_full")
        if tl is not None:
            payload["Tl"] = tl
        if tg is not None:
            payload["Tg"] = tg
        if yl is not None:
            payload["Yl"] = yl
        if yg is not None:
            payload["Yg"] = yg
        centers = _grid_centers(grid)
        faces = _grid_faces(grid)
        if centers is not None:
            payload["r_c"] = centers
        if faces is not None:
            payload["r_f"] = faces
        meta = _grid_meta(grid)
        for key, value in meta.items():
            if value != "":
                payload["Nl" if key == "n_liq" else "Ng" if key == "n_gas" else "iface_f"] = int(value)
        for src_name, out_name in (("rho_l", "rho_l"), ("rho_g", "rho_g"), ("h_l", "h_l"), ("h_g", "h_g"), ("hl", "h_l"), ("hg", "h_g")):
            value = getattr(props, src_name, None) if props is not None else None
            if value is None:
                continue
            arr = np.asarray(value, dtype=np.float64)
            if out_name not in payload:
                payload[out_name] = arr

    if u_raw is not None:
        payload["u_raw"] = np.asarray(u_raw, dtype=np.float64)
    elif layout is not None and getattr(layout, "total_size", None) is not None:
        payload["layout_total_size"] = int(layout.total_size)

    if state is None and u_raw is None:
        payload["snapshot_missing"] = np.int64(1)
    elif "layout_total_size" not in payload and layout is not None and getattr(layout, "total_size", None) is not None:
        payload["layout_total_size"] = int(layout.total_size)

    np.savez_compressed(out_path, **payload)


def _minimal_mapping_payload(*, output_layout: OutputLayout, grid: object | None, layout: object | None, cfg: object | None) -> dict[str, Any]:
    meta = _grid_meta(grid)
    return {
        "case_id": output_layout.case_id,
        "run_id": output_layout.run_id,
        "layout_total_size": "" if layout is None or getattr(layout, "total_size", None) is None else int(layout.total_size),
        "species_order": _species_order_payload(cfg),
        "grid": {
            "n_liq": meta.get("n_liq", ""),
            "n_gas": meta.get("n_gas", ""),
            "iface_f": meta.get("iface_f", ""),
        },
        "fields": {
            "Ts": {"kind": "scalar"},
            "Rd": {"kind": "scalar"},
            "mpp": {"kind": "scalar"},
            "Tl": {"kind": "cell_field"},
            "Tg": {"kind": "cell_field"},
            "Yl": {"kind": "cell_species"},
            "Yg": {"kind": "cell_species"},
            "r_c": {"kind": "cell_coordinate"},
            "r_f": {"kind": "face_coordinate"},
            "u_raw": {"kind": "layout_vector"},
        },
        "conventions": {
            "mpp_sign": "positive_outward_from_liquid",
            "radial_coordinate": "spherical_radius",
        },
    }


def _write_failure_mapping(
    flayout: FailureLayout,
    *,
    output_layout: OutputLayout,
    grid: object | None = None,
    layout: object | None = None,
    cfg: object | None = None,
) -> None:
    if _copy_if_exists(output_layout.mapping_json, flayout.mapping_json):
        return
    _write_json(
        flayout.mapping_json,
        _minimal_mapping_payload(output_layout=output_layout, grid=grid, layout=layout, cfg=cfg),
    )


def _copy_runtime_log_tails(
    flayout: FailureLayout,
    *,
    output_layout: OutputLayout,
    copy_text_log: bool,
    copy_jsonl_events: bool,
    text_lines: int,
    jsonl_lines: int,
) -> dict[str, Any]:
    result = {
        "has_run_log": bool(output_layout.run_log.exists()),
        "has_run_events_jsonl": bool(output_layout.run_events_jsonl.exists()),
        "has_run_log_tail": False,
        "has_run_events_tail": False,
        "run_log": str(output_layout.run_log),
        "run_events_jsonl": str(output_layout.run_events_jsonl),
        "run_log_tail": str(flayout.run_log_tail_txt),
        "run_events_tail_jsonl": str(flayout.run_events_tail_jsonl),
    }

    if copy_text_log and output_layout.run_log.exists():
        try:
            copy_log_tail(output_layout.run_log, flayout.run_log_tail_txt, n_lines=int(text_lines))
            result["has_run_log_tail"] = bool(flayout.run_log_tail_txt.exists())
        except Exception as exc:
            result["run_log_tail_error"] = f"{exc.__class__.__name__}: {exc}"

    if copy_jsonl_events and output_layout.run_events_jsonl.exists():
        try:
            copy_jsonl_tail(output_layout.run_events_jsonl, flayout.run_events_tail_jsonl, n_lines=int(jsonl_lines))
            result["has_run_events_tail"] = bool(flayout.run_events_tail_jsonl.exists())
        except Exception as exc:
            result["run_events_tail_error"] = f"{exc.__class__.__name__}: {exc}"

    return result


def write_rank_failure_meta(
    flayout: FailureLayout,
    *,
    rank_id: int,
    payload: Mapping[str, Any],
) -> Path:
    out = failure_rank_meta_path(flayout, rank=int(rank_id))
    meta_payload = {str(k): v for k, v in payload.items() if str(k) not in {"x_owned", "snes_last_x_owned"}}
    meta_payload["rank_id"] = int(rank_id)
    _write_json(out, meta_payload)
    return out


def write_rank_last_x_csv(
    flayout: FailureLayout,
    *,
    rank_id: int,
    x_owned,
    global_range=None,
    tag: str = "snes_last_x",
) -> Path | None:
    if x_owned is None:
        return None
    values = np.asarray(x_owned, dtype=np.float64).reshape(-1)
    if str(tag) == "snes_last_x":
        out = failure_rank_snes_last_x_csv_path(flayout, rank=int(rank_id))
    else:
        out = flayout.root / f"rank_{int(rank_id):03d}_{str(tag)}.csv"
    _ensure_parent(out)
    rstart = None
    if global_range is not None:
        global_range_tuple = tuple(global_range)
        if len(global_range_tuple) >= 1:
            rstart = int(global_range_tuple[0])
    with out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["global_idx", "value"])
        for i, value in enumerate(values):
            global_idx = i if rstart is None else rstart + i
            writer.writerow([global_idx, value])
    return out


def _iter_rank_payloads(
    ctx_meta: Mapping[str, Any] | None,
    rank_failure_meta: object | None,
) -> tuple[tuple[int, dict[str, Any]], ...]:
    payloads: list[tuple[int, dict[str, Any]]] = []
    if isinstance(rank_failure_meta, Mapping):
        if any(key in rank_failure_meta for key in ("rank_id", "x_owned", "snes_last_x_owned")):
            rank_id = int(rank_failure_meta.get("rank_id", 0))
            payloads.append((rank_id, dict(rank_failure_meta)))
        else:
            for key, value in rank_failure_meta.items():
                if isinstance(value, Mapping):
                    rank_id = int(value.get("rank_id", key))
                    payloads.append((rank_id, dict(value)))
    if payloads:
        return tuple(payloads)
    if isinstance(ctx_meta, Mapping):
        candidate_keys = {
            "rank",
            "rank_id",
            "snes_last_x_owned",
            "snes_last_x_range",
            "snes_last_x_nonfinite",
            "snes_last_x_phase",
            "snes_last_x_min",
            "snes_last_x_max",
            "snes_last_global",
            "penalty_last_reason",
            "penalty_last_stage",
            "ls_postcheck_last",
            "local_dof_range",
            "f_idx_global",
            "f_desc",
        }
        extracted = {k: v for k, v in ctx_meta.items() if k in candidate_keys}
        if extracted:
            rank_id = int(extracted.get("rank_id", extracted.get("rank", 0)))
            payloads.append((rank_id, extracted))
    return tuple(payloads)


def _write_rank_sidecars(
    flayout: FailureLayout,
    *,
    ctx_meta: Mapping[str, Any] | None,
    rank_failure_meta: object | None,
    write_meta: bool,
    write_last_x: bool,
) -> None:
    for rank_id, payload in _iter_rank_payloads(ctx_meta, rank_failure_meta):
        if write_meta:
            write_rank_failure_meta(flayout, rank_id=rank_id, payload=payload)
        if write_last_x:
            x_owned = payload.get("snes_last_x_owned", payload.get("x_owned"))
            global_range = payload.get("snes_last_x_range", payload.get("global_range"))
            write_rank_last_x_csv(
                flayout,
                rank_id=rank_id,
                x_owned=x_owned,
                global_range=global_range,
                tag=str(payload.get("tag", "snes_last_x")),
            )


def write_attempt_failure(
    *,
    cfg,
    output_layout,
    step_id: int,
    retry_id: int,
    t_old: float,
    dt_try: float,
    failure_class: str,
    reject_reason: str,
    state_last_good=None,
    state_attempt_input=None,
    state_attempt_final=None,
    u_last_good=None,
    u_attempt_input=None,
    u_attempt_final=None,
    diag=None,
    runtime=None,
    grid=None,
    layout=None,
    props_last_good=None,
    props_attempt_input=None,
    props_attempt_final=None,
    ctx_meta=None,
    rank_failure_meta=None,
    is_root: bool | None = None,
) -> FailureLayout | None:
    if not _is_root_writer(runtime=runtime, is_root=is_root):
        return None

    write_report = _cfg_flag(cfg, "diagnostics", "write_failure_report", True)
    write_rank_meta = _cfg_flag(cfg, "diagnostics", "write_rank_failure_meta", True)
    write_rank_last_x = _cfg_flag(cfg, "diagnostics", "write_rank_last_x", True)
    write_snapshots = _cfg_flag(cfg, "output", "write_failure_snapshot", True)
    copy_tail_to_failure = _cfg_flag(cfg, "diagnostics", "runtime_log_copy_tail_to_failure", True)
    failure_log_tail_lines = _cfg_int(cfg, "diagnostics", "failure_log_tail_lines", 200)
    failure_events_tail_lines = _cfg_int(cfg, "diagnostics", "failure_events_tail_lines", 200)

    flayout = build_failure_layout(output_layout, step_id=step_id, retry_id=retry_id)
    ensure_failure_dir(flayout)
    log_tail_meta = _copy_runtime_log_tails(
        flayout,
        output_layout=output_layout,
        copy_text_log=bool(copy_tail_to_failure),
        copy_jsonl_events=bool(copy_tail_to_failure),
        text_lines=int(failure_log_tail_lines),
        jsonl_lines=int(failure_events_tail_lines),
    )

    if write_report:
        report = _build_failure_report_payload(
            step_id=step_id,
            retry_id=retry_id,
            t_old=t_old,
            dt_try=dt_try,
            failure_class=failure_class,
            reject_reason=reject_reason,
            diag=diag,
            ctx_meta=ctx_meta if isinstance(ctx_meta, Mapping) else None,
            state_attempt_final=state_attempt_final,
            log_tail_meta=log_tail_meta,
            output_layout=output_layout,
            flayout=flayout,
        )
        report["has_state_last_good"] = bool(state_last_good is not None)
        report["has_state_attempt_input"] = bool(state_attempt_input is not None)
        report["has_state_attempt_final"] = bool(state_attempt_final is not None)
        _write_json(flayout.report_json, report)

    if write_snapshots:
        _write_failure_state_snapshot(
            flayout.state_last_good_npz,
            label="last_good",
            step_id=step_id,
            retry_id=retry_id,
            t=t_old,
            dt_try=dt_try,
            state=state_last_good,
            u_raw=u_last_good,
            grid=grid,
            props=props_last_good,
            layout=layout,
        )
        _write_failure_state_snapshot(
            flayout.state_attempt_input_npz,
            label="attempt_input",
            step_id=step_id,
            retry_id=retry_id,
            t=t_old,
            dt_try=dt_try,
            state=state_attempt_input,
            u_raw=u_attempt_input,
            grid=grid,
            props=props_attempt_input,
            layout=layout,
        )
        _write_failure_state_snapshot(
            flayout.state_attempt_final_npz,
            label="attempt_final",
            step_id=step_id,
            retry_id=retry_id,
            t=t_old,
            dt_try=dt_try,
            state=state_attempt_final,
            u_raw=u_attempt_final,
            grid=grid,
            props=props_attempt_final,
            layout=layout,
        )

    _write_failure_mapping(
        flayout,
        output_layout=output_layout,
        grid=grid,
        layout=layout,
        cfg=cfg,
    )
    _write_rank_sidecars(
        flayout,
        ctx_meta=ctx_meta if isinstance(ctx_meta, Mapping) else None,
        rank_failure_meta=rank_failure_meta,
        write_meta=write_rank_meta,
        write_last_x=write_rank_last_x,
    )
    return flayout


__all__ = [
    "FailureLayout",
    "OutputLayout",
    "write_attempt_failure",
    "write_rank_failure_meta",
    "write_rank_last_x_csv",
]
