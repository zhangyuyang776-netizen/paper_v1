from __future__ import annotations

"""Runtime state container for the evaporation-only Phase 7 driver path."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping
import importlib.util
import sys
import time

from core.layout import UnknownLayout
import core.types as core_types


CaseConfig = getattr(core_types, "CaseConfig", core_types.RunConfig)
Grid1D = getattr(core_types, "Grid1D", core_types.Mesh1D)
Props = core_types.Props
State = core_types.State


def _load_output_layout_symbols():
    module_path = Path(__file__).resolve().parents[1] / "io" / "output_layout.py"
    spec = importlib.util.spec_from_file_location("paper_v1_io_output_layout", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("failed to load io/output_layout.py")
    module = sys.modules.get(spec.name)
    if module is None:
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    return module.OutputLayout, module.build_output_layout, module.ensure_output_dirs


OutputLayout, build_output_layout, ensure_output_dirs = _load_output_layout_symbols()


def _load_runtime_logging_module():
    module_path = Path(__file__).resolve().parents[1] / "io" / "runtime_logging.py"
    spec = importlib.util.spec_from_file_location("paper_v1_io_runtime_logging", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("failed to load io/runtime_logging.py")
    module = sys.modules.get(spec.name)
    if module is None:
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    return module

_BOUND_OUTPUT_LAYOUTS: dict[int, OutputLayout] = {}
_BOUND_RUNTIMES: dict[int, object] = {}


@dataclass(slots=True)
class RunCounters:
    n_steps_attempted: int = 0
    n_steps_accepted: int = 0
    n_retries_total: int = 0
    n_failures_total: int = 0


@dataclass(slots=True)
class RuntimeEVP:
    cfg: CaseConfig

    run_id: str
    created_at: str
    schema_version: str

    backend: str
    petsc_enabled: bool
    parallel_mode: str
    single_rank_only_boundary_active: bool

    rank: int
    size: int
    is_root: bool

    output_layout: OutputLayout
    parallel_handles: object | None
    logger: object | None

    layout: UnknownLayout
    grid: Grid1D
    state: State
    props: Props

    t: float
    next_step_id: int

    mapping_written: bool = False
    last_step_result: object | None = None
    last_inner_result: object | None = None
    last_dt_used: float | None = None
    last_message: str = ""

    ended_by: str = ""
    exit_code: int | None = None

    wallclock_start: float = 0.0
    wallclock_end: float | None = None

    counters: RunCounters = field(default_factory=RunCounters)

    @property
    def mpi_size(self) -> int:
        return int(self.size)


def _cfg_case_id_for_layout(cfg: object) -> str:
    case_section = getattr(cfg, "case", None)
    if case_section is not None:
        case_id = getattr(case_section, "case_id", getattr(case_section, "id", None))
        if case_id is not None:
            return str(case_id)
    case_name = getattr(cfg, "case_name", None)
    if case_name is not None:
        return str(case_name)
    raise ValueError("cfg must expose case.case_id, case.id, or case_name for output layout construction")


def _cfg_for_output_layout(cfg: object):
    if getattr(cfg, "case", None) is not None:
        return cfg
    return SimpleNamespace(
        case=SimpleNamespace(case_id=_cfg_case_id_for_layout(cfg)),
        paths=getattr(cfg, "paths", None),
    )


def _safe_setattr(obj: object, name: str, value: object) -> bool:
    try:
        setattr(obj, name, value)
        return True
    except Exception:
        return False


def _safe_delattr(obj: object, name: str) -> None:
    try:
        delattr(obj, name)
    except Exception:
        pass


def _bool_is_root(rank: int) -> bool:
    return int(rank) == 0


def _wallclock_seconds(runtime: RuntimeEVP) -> float | None:
    if runtime.wallclock_end is None:
        return None
    return max(0.0, float(runtime.wallclock_end) - float(runtime.wallclock_start))


def _build_runtime_logger(output_layout, cfg, *, rank: int, is_root: bool):
    runtime_logging = _load_runtime_logging_module()
    return runtime_logging.build_runtime_logger(
        output_layout=output_layout,
        cfg=cfg,
        rank=rank,
        is_root=is_root,
    )


def _payload_get(payload: object, key: str, default: Any = None) -> Any:
    if isinstance(payload, Mapping):
        return payload.get(key, default)
    return getattr(payload, key, default)


def _payload_to_dict(payload: object | None) -> dict[str, Any]:
    if payload is None:
        return {}
    if isinstance(payload, Mapping):
        return {str(k): v for k, v in payload.items()}
    if hasattr(payload, "__dict__"):
        return {str(k): v for k, v in vars(payload).items() if not str(k).startswith("_")}
    return {"value": payload}


def _payload_level(payload_dict: Mapping[str, Any], default: str = "INFO") -> str:
    level = str(payload_dict.get("level", default)).strip().upper()
    if level not in {"DEBUG", "INFO", "WARNING", "ERROR"}:
        return str(default).strip().upper()
    return level


def _failure_class_value(value: Any) -> Any:
    if value is None:
        return None
    return getattr(value, "value", value)


def make_run_id(*, prefix: str | None = None, now: datetime | None = None) -> str:
    dt = now or datetime.now(timezone.utc)
    stamp = dt.strftime("%Y%m%d_%H%M%S")
    token = f"{prefix}_{stamp}" if prefix else stamp
    return token.replace(" ", "_")


def make_created_at(now: datetime | None = None) -> str:
    dt = now or datetime.now(timezone.utc)
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def build_runtime_evp(
    cfg: CaseConfig,
    *,
    layout: UnknownLayout,
    grid0: Grid1D,
    state0: State,
    props0: Props,
    t0: float,
    backend: str,
    petsc_enabled: bool,
    parallel_mode: str,
    rank: int,
    size: int,
    parallel_handles: object | None = None,
    single_rank_only_boundary_active: bool = False,
    run_id: str | None = None,
    created_at: str | None = None,
    schema_version: str = "phase7_evp_v1",
    step_id0: int = 0,
) -> RuntimeEVP:
    rid = run_id or make_run_id()
    created = created_at or make_created_at()
    layout_cfg = _cfg_for_output_layout(cfg)
    out_layout = build_output_layout(layout_cfg, run_id=rid)
    ensure_output_dirs(out_layout)
    is_root = _bool_is_root(int(rank))
    logger = _build_runtime_logger(out_layout, cfg, rank=int(rank), is_root=is_root)
    return RuntimeEVP(
        cfg=cfg,
        run_id=rid,
        created_at=created,
        schema_version=str(schema_version),
        backend=str(backend),
        petsc_enabled=bool(petsc_enabled),
        parallel_mode=str(parallel_mode),
        single_rank_only_boundary_active=bool(single_rank_only_boundary_active),
        rank=int(rank),
        size=int(size),
        is_root=is_root,
        output_layout=out_layout,
        parallel_handles=parallel_handles,
        logger=logger,
        layout=layout,
        grid=grid0,
        state=state0,
        props=props0,
        t=float(t0),
        next_step_id=int(step_id0),
        wallclock_start=time.perf_counter(),
    )


def bind_cfg_output_layout(cfg: CaseConfig, runtime: RuntimeEVP) -> None:
    bound = _safe_setattr(cfg, "_output_layout", runtime.output_layout)
    if not bound:
        _BOUND_OUTPUT_LAYOUTS[id(cfg)] = runtime.output_layout
    paths = getattr(cfg, "paths", None)
    if paths is not None:
        _safe_setattr(paths, "_output_layout", runtime.output_layout)


def bind_cfg_runtime_public(cfg: CaseConfig, runtime: RuntimeEVP) -> None:
    bound = _safe_setattr(cfg, "runtime", runtime)
    if not bound:
        _BOUND_RUNTIMES[id(cfg)] = runtime


def get_bound_output_layout(cfg: object) -> OutputLayout | None:
    return _BOUND_OUTPUT_LAYOUTS.get(id(cfg))


def get_bound_runtime(cfg: object) -> object | None:
    return _BOUND_RUNTIMES.get(id(cfg))


def unbind_cfg_runtime_bridges(cfg: CaseConfig) -> None:
    _safe_delattr(cfg, "_output_layout")
    _safe_delattr(cfg, "runtime")
    paths = getattr(cfg, "paths", None)
    if paths is not None:
        _safe_delattr(paths, "_output_layout")
    _BOUND_OUTPUT_LAYOUTS.pop(id(cfg), None)
    _BOUND_RUNTIMES.pop(id(cfg), None)


def note_attempt(runtime: RuntimeEVP, *, is_retry: bool) -> None:
    runtime.counters.n_steps_attempted += 1
    if is_retry:
        runtime.counters.n_retries_total += 1


def commit_accepted_step(
    runtime: RuntimeEVP,
    *,
    step_result,
    t_new: float,
    dt_used: float,
    layout_new: UnknownLayout | None = None,
    grid_new: Grid1D | None = None,
    state_new: State | None = None,
    props_new: Props | None = None,
    last_inner_result: object | None = None,
) -> None:
    next_layout = layout_new if layout_new is not None else getattr(step_result, "layout_new", None)
    next_grid = grid_new if grid_new is not None else getattr(step_result, "grid_new", None)
    next_state = state_new if state_new is not None else getattr(step_result, "state_new", None)
    next_props = props_new if props_new is not None else getattr(step_result, "props_new", None)

    if next_layout is not None:
        runtime.layout = next_layout
    if next_grid is not None:
        runtime.grid = next_grid
    if next_state is not None:
        runtime.state = next_state
    if next_props is not None:
        runtime.props = next_props

    runtime.t = float(t_new)
    runtime.next_step_id += 1
    runtime.counters.n_steps_accepted += 1
    runtime.last_step_result = step_result
    runtime.last_inner_result = last_inner_result
    runtime.last_dt_used = float(dt_used)
    runtime.last_message = str(getattr(step_result, "message", "") or "")


def record_run_end(runtime: RuntimeEVP, *, exit_code: int, ended_by: str, message: str = "") -> None:
    runtime.exit_code = int(exit_code)
    runtime.ended_by = str(ended_by)
    runtime.last_message = str(message)
    runtime.wallclock_end = time.perf_counter()


def build_run_summary_payload(runtime: RuntimeEVP) -> dict[str, Any]:
    return {
        "run_id": runtime.run_id,
        "case_id": runtime.output_layout.case_id,
        "exit_code": runtime.exit_code,
        "ended_by": runtime.ended_by,
        "message": runtime.last_message,
        "n_steps_attempted": runtime.counters.n_steps_attempted,
        "n_steps_accepted": runtime.counters.n_steps_accepted,
        "n_retries_total": runtime.counters.n_retries_total,
        "n_failures_total": runtime.counters.n_failures_total,
        "t_final": runtime.t,
        "dt_last": runtime.last_dt_used,
        "wallclock_seconds": _wallclock_seconds(runtime),
        "backend": runtime.backend,
        "mpi_size": runtime.size,
    }


def runtime_log(runtime: RuntimeEVP, level: str, message: str, **fields: Any) -> None:
    logger = getattr(runtime, "logger", None)
    if logger is None:
        return
    runtime_logging = _load_runtime_logging_module()
    runtime_logging.log_text(logger, level, message, **fields)


def runtime_event(
    runtime: RuntimeEVP,
    event: str,
    payload: Mapping[str, Any] | None = None,
    *,
    level: str = "INFO",
) -> None:
    logger = getattr(runtime, "logger", None)
    if logger is None:
        return
    runtime_logging = _load_runtime_logging_module()
    runtime_logging.log_event(logger, event, payload, level=level)


def runtime_exception(
    runtime: RuntimeEVP,
    where: str,
    exc: BaseException,
    *,
    step_id: int | None = None,
    outer_iter: int | None = None,
    stage: str | None = None,
    extra: Mapping[str, Any] | None = None,
) -> None:
    logger = getattr(runtime, "logger", None)
    if logger is None:
        return
    runtime_logging = _load_runtime_logging_module()
    runtime_logging.log_exception(
        logger,
        where,
        exc,
        step_id=step_id,
        outer_iter=outer_iter,
        stage=stage,
        extra=extra,
    )


def make_diag_sink(runtime: RuntimeEVP):
    def _sink(*, event: str, payload: object | None = None) -> None:
        payload_dict = _payload_to_dict(payload)
        event_level = _payload_level(payload_dict, "INFO")
        runtime_event(runtime, event, payload_dict, level=event_level)

        logger = getattr(runtime, "logger", None)
        if logger is None:
            return

        step_id = _payload_get(payload, "step_id", payload_dict.get("step_id"))
        outer_iter = _payload_get(
            payload,
            "outer_iter_index",
            _payload_get(payload, "outer_iter", payload_dict.get("outer_iter")),
        )

        if event == "predictor_ready":
            if bool(getattr(logger.cfg, "outer_detail", True)):
                runtime_log(
                    runtime,
                    event_level,
                    "predictor_ready",
                    step_id=step_id,
                    a_pred=_payload_get(payload, "a_pred", payload_dict.get("a_pred")),
                    dot_a_pred=_payload_get(payload, "dot_a_pred", payload_dict.get("dot_a_pred")),
                    a_old=_payload_get(payload, "a_old", payload_dict.get("a_old")),
                    dot_a_old=_payload_get(payload, "dot_a_old", payload_dict.get("dot_a_old")),
                )
            return

        if event == "outer_iter_begin":
            if bool(getattr(logger.cfg, "outer_detail", True)):
                runtime_log(
                    runtime,
                    event_level,
                    "outer_iter_begin",
                    step_id=step_id,
                    outer_iter=outer_iter,
                    a_iter=_payload_get(payload, "a_iter", payload_dict.get("a_iter")),
                    dot_a_iter=_payload_get(payload, "dot_a_iter", payload_dict.get("dot_a_iter")),
                )
            return

        if event == "outer_iteration":
            if bool(getattr(logger.cfg, "outer_detail", True)):
                inner = _payload_get(payload, "inner", None)
                convergence = _payload_get(payload, "convergence", None)
                stats = _payload_get(inner, "stats", None)
                residuals = _payload_get(stats, "residual_norms", None)
                runtime_log(
                    runtime,
                    event_level,
                    "outer_iteration",
                    step_id=step_id,
                    outer_iter=outer_iter,
                    inner_converged=_payload_get(inner, "converged", None),
                    inner_iter=_payload_get(stats, "inner_iter_count", None),
                    linear_iter=_payload_get(stats, "linear_iter_count", None),
                    Rinf=_payload_get(residuals, "linf", None),
                    snes_reason=_payload_get(stats, "snes_reason", None),
                    ksp_reason=_payload_get(stats, "ksp_reason", None),
                    eps_dot_a=_payload_get(convergence, "eps_dot_a", None),
                    outer_converged=_payload_get(convergence, "converged", None),
                )
            return

        if event == "outer_inner_result":
            if bool(getattr(logger.cfg, "inner_summary", True)) or bool(getattr(logger.cfg, "outer_detail", True)):
                runtime_log(
                    runtime,
                    event_level,
                    "outer_inner_result",
                    step_id=step_id,
                    outer_iter=outer_iter,
                    inner_converged=_payload_get(payload, "inner_converged", payload_dict.get("inner_converged")),
                    inner_iter=_payload_get(payload, "inner_iter_count", payload_dict.get("inner_iter_count")),
                    linear_iter=_payload_get(payload, "linear_iter_count", payload_dict.get("linear_iter_count")),
                    Rl2=_payload_get(payload, "residual_l2", payload_dict.get("residual_l2")),
                    Rinf=_payload_get(payload, "residual_linf", payload_dict.get("residual_linf")),
                    snes_reason=_payload_get(payload, "snes_reason", payload_dict.get("snes_reason")),
                    ksp_reason=_payload_get(payload, "ksp_reason", payload_dict.get("ksp_reason")),
                    dot_a_phys=_payload_get(payload, "dot_a_phys", payload_dict.get("dot_a_phys")),
                )
            return

        if event == "outer_convergence_result":
            if bool(getattr(logger.cfg, "outer_detail", True)):
                runtime_log(
                    runtime,
                    event_level,
                    "outer_convergence_result",
                    step_id=step_id,
                    outer_iter=outer_iter,
                    eps_dot_a=_payload_get(payload, "eps_dot_a", payload_dict.get("eps_dot_a")),
                    converged=_payload_get(payload, "converged", payload_dict.get("converged")),
                    dot_a_iter=_payload_get(payload, "dot_a_iter", payload_dict.get("dot_a_iter")),
                    dot_a_phys=_payload_get(payload, "dot_a_phys", payload_dict.get("dot_a_phys")),
                )
            return

        if event == "outer_corrector_result":
            if bool(getattr(logger.cfg, "outer_detail", True)):
                runtime_log(
                    runtime,
                    event_level,
                    "outer_corrector_result",
                    step_id=step_id,
                    outer_iter=outer_iter,
                    a_new=_payload_get(payload, "a_new", payload_dict.get("a_new")),
                    dot_a_new=_payload_get(payload, "dot_a_new", payload_dict.get("dot_a_new")),
                )
            return

        if event == "step_acceptance_decision":
            runtime_log(
                runtime,
                event_level,
                "step_acceptance_decision",
                step_id=step_id,
                accepted=_payload_get(payload, "accepted", payload_dict.get("accepted")),
                action=_payload_get(payload, "action", payload_dict.get("action")),
                dt_current=_payload_get(payload, "dt_current", payload_dict.get("dt_current")),
                dt_next=_payload_get(payload, "dt_next", payload_dict.get("dt_next")),
                retries_used=_payload_get(payload, "retries_used", payload_dict.get("retries_used")),
                reject_reason=_payload_get(payload, "reject_reason", payload_dict.get("reject_reason")),
                failure_class=_failure_class_value(_payload_get(payload, "failure_class", payload_dict.get("failure_class"))),
            )
            return

        if event == "step_retry":
            diagnostics = _payload_get(payload, "diagnostics", {}) or {}
            decision = _payload_get(payload, "acceptance", None)
            failure = _payload_get(payload, "failure", None)
            runtime_log(
                runtime,
                event_level if event_level != "INFO" else "WARNING",
                "step_retry",
                step_id=step_id,
                old_dt=_payload_get(payload, "dt_attempt", payload_dict.get("dt_attempt")),
                new_dt=_payload_get(decision, "dt_next", None),
                retries_used=_payload_get(diagnostics, "retries_used", None),
                reject_reason=_payload_get(decision, "reject_reason", _payload_get(payload, "reject_reason", None)),
                failure_class=_failure_class_value(_payload_get(failure, "failure_class", None)),
            )
            return

        if event == "step_result":
            diagnostics = _payload_get(payload, "diagnostics", {}) or {}
            accepted = bool(_payload_get(payload, "accepted", False))
            decision = _payload_get(payload, "acceptance", None)
            if accepted:
                accepted_geometry = _payload_get(payload, "accepted_geometry", None)
                runtime_log(
                    runtime,
                    event_level,
                    "step_result accepted",
                    step_id=step_id,
                    retries_used=_payload_get(diagnostics, "retries_used", None),
                    Rd=_payload_get(accepted_geometry, "a", None),
                    Ts=_payload_get(payload, "Ts", None),
                    mpp=_payload_get(payload, "mpp", None),
                )
            else:
                failure = _payload_get(payload, "failure", None)
                runtime_log(
                    runtime,
                    event_level if event_level != "INFO" else "WARNING",
                    "step_result rejected",
                    step_id=step_id,
                    retries_used=_payload_get(diagnostics, "retries_used", None),
                    failure_class=_failure_class_value(_payload_get(failure, "failure_class", None)),
                    reject_reason=_payload_get(decision, "reject_reason", _payload_get(payload, "reject_reason", None)),
                )
            return

        if event == "outer_exception":
            runtime_log(
                runtime,
                event_level if event_level != "INFO" else "ERROR",
                "outer_exception",
                step_id=step_id,
                outer_iter=outer_iter,
                stage=_payload_get(payload, "stage", None),
                exc=_payload_get(payload, "exception_type", None),
                msg=_payload_get(payload, "message", None),
            )
            return

        if event == "inner_solve_begin":
            if bool(getattr(logger.cfg, "inner_summary", True)):
                runtime_log(
                    runtime,
                    event_level,
                    "inner_solve_begin",
                    step_id=step_id,
                    outer_iter=outer_iter,
                    N=_payload_get(payload, "N", payload_dict.get("N")),
                    jacobian_mode=_payload_get(payload, "jacobian_mode", payload_dict.get("jacobian_mode")),
                    snes_type=_payload_get(payload, "snes_type", payload_dict.get("snes_type")),
                    linesearch_type=_payload_get(payload, "linesearch_type", payload_dict.get("linesearch_type")),
                    ksp_type=_payload_get(payload, "ksp_type", payload_dict.get("ksp_type")),
                    pc_type=_payload_get(payload, "pc_type", payload_dict.get("pc_type")),
                    comm_size=_payload_get(payload, "comm_size", payload_dict.get("comm_size")),
                    parallel_active=_payload_get(payload, "parallel_active", payload_dict.get("parallel_active")),
                )
            return

        if event == "inner_monitor":
            if not bool(getattr(logger.cfg, "inner_monitor", False)):
                return
            runtime_log(
                runtime,
                event_level,
                "inner_monitor",
                step_id=step_id,
                outer_iter=outer_iter,
                snes_iter=_payload_get(payload, "snes_iter", payload_dict.get("snes_iter")),
                fnorm=_payload_get(payload, "fnorm", payload_dict.get("fnorm")),
                res_inf_eval=_payload_get(payload, "res_inf_eval", payload_dict.get("res_inf_eval")),
                res_inf_phys=_payload_get(payload, "res_inf_phys", payload_dict.get("res_inf_phys")),
                residual_linf=_payload_get(payload, "residual_linf", payload_dict.get("residual_linf")),
            )
            return

        if event == "inner_solve_end":
            if bool(getattr(logger.cfg, "inner_summary", True)):
                converged = _payload_get(payload, "converged", payload_dict.get("converged"))
                level_eff = event_level if "level" in payload_dict else ("INFO" if bool(converged) else "WARNING")
                runtime_log(
                    runtime,
                    level_eff,
                    "inner_solve_end",
                    step_id=step_id,
                    outer_iter=outer_iter,
                    converged=converged,
                    inner_iter=_payload_get(payload, "inner_iter_count", payload_dict.get("inner_iter_count")),
                    linear_iter=_payload_get(payload, "linear_iter_count", payload_dict.get("linear_iter_count")),
                    Rl2=_payload_get(payload, "residual_l2", payload_dict.get("residual_l2")),
                    Rinf=_payload_get(payload, "residual_linf", payload_dict.get("residual_linf")),
                    snes_reason=_payload_get(payload, "snes_reason", payload_dict.get("snes_reason")),
                    ksp_reason=_payload_get(payload, "ksp_reason", payload_dict.get("ksp_reason")),
                    dot_a_phys=_payload_get(payload, "dot_a_phys", payload_dict.get("dot_a_phys")),
                    time_total=_payload_get(payload, "time_total", payload_dict.get("time_total")),
                    jacobian_mode=_payload_get(payload, "jacobian_mode", payload_dict.get("jacobian_mode")),
                )
            return

        if event == "inner_solve_failure":
            level_eff = event_level if "level" in payload_dict else "ERROR"
            runtime_log(
                runtime,
                level_eff,
                "inner_solve_failure",
                step_id=step_id,
                outer_iter=outer_iter,
                exc=_payload_get(payload, "exception_type", payload_dict.get("exception_type")),
                msg=_payload_get(payload, "message", payload_dict.get("message")),
                snes_reason=_payload_get(payload, "snes_reason", payload_dict.get("snes_reason")),
                ksp_reason=_payload_get(payload, "ksp_reason", payload_dict.get("ksp_reason")),
                last_res_inf_eval=_payload_get(payload, "last_res_inf_eval", payload_dict.get("last_res_inf_eval")),
                last_res_inf_phys=_payload_get(payload, "last_res_inf_phys", payload_dict.get("last_res_inf_phys")),
                n_func_eval=_payload_get(payload, "n_func_eval", payload_dict.get("n_func_eval")),
                n_jac_eval=_payload_get(payload, "n_jac_eval", payload_dict.get("n_jac_eval")),
                time_inner_wall=_payload_get(payload, "time_inner_wall", payload_dict.get("time_inner_wall")),
                jacobian_mode=_payload_get(payload, "jacobian_mode", payload_dict.get("jacobian_mode")),
            )
            return

        if bool(getattr(logger.cfg, "step_summary", True)):
            runtime_log(runtime, event_level, event, step_id=step_id, outer_iter=outer_iter)

    return _sink


__all__ = [
    "RunCounters",
    "RuntimeEVP",
    "bind_cfg_output_layout",
    "bind_cfg_runtime_public",
    "build_run_summary_payload",
    "build_runtime_evp",
    "get_bound_output_layout",
    "get_bound_runtime",
    "make_created_at",
    "make_diag_sink",
    "make_run_id",
    "commit_accepted_step",
    "note_attempt",
    "record_run_end",
    "runtime_event",
    "runtime_exception",
    "runtime_log",
    "unbind_cfg_runtime_bridges",
]
