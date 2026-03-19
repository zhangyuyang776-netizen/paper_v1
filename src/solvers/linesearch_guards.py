from __future__ import annotations

"""Domain/admissibility guards for PETSc line-search driven inner solves.

This module provides a small, structured guard layer for:
- cheap trial-vector legality checks before/around line search
- trial-physics snapshot legality checks after residual-side evaluation
- structured failure/diagnostic objects for solver and step-level consumers

It intentionally does not perform remap, recovery, property evaluation,
interface equilibrium, residual assembly, or Jacobian assembly.
"""

from dataclasses import dataclass, field
from math import isfinite
from typing import Any

import numpy as np

from .nonlinear_context import NonlinearContext
from .nonlinear_types import FailureClass, FailureInfo, GuardFailureReason


@dataclass(slots=True, kw_only=True)
class LineSearchGuardConfig:
    """Configuration controlling vector/snapshot guard behavior.

    Built from ``NonlinearContext.cfg`` plus optional context metadata.
    Consumed by pure guard checks and PETSc pre/post-check adapter factories.
    """

    enabled: bool = True
    enable_precheck: bool = True
    enable_postcheck: bool = True

    allow_small_ts_clip: bool = True
    ts_clip_abs_tol: float = 1.0e-8
    ts_hard_min: float | None = None
    ts_hard_max: float | None = None

    allow_small_mpp_clip: bool = False
    mpp_clip_abs_tol: float = 0.0
    mpp_hard_min: float | None = None
    mpp_hard_max: float | None = None

    temperature_hard_min: float | None = None
    temperature_hard_max: float | None = None
    composition_hard_tol: float = 1.0e-12
    density_min: float = 1.0e-300
    require_finite_state: bool = True
    require_finite_residual: bool = True


@dataclass(slots=True, kw_only=True)
class TrialPhysicsSnapshot:
    """Lightweight trial-physics snapshot consumed by post-check guards.

    Produced by residual/SNES wrapper code and consumed here without
    reconstructing a second physics/property pipeline.
    """

    Ts: float | None = None
    mpp: float | None = None

    Tg_min: float | None = None
    Tg_max: float | None = None
    Tl_min: float | None = None
    Tl_max: float | None = None

    rho_g_min: float | None = None
    rho_l_min: float | None = None

    y_g_min: float | None = None
    y_g_max: float | None = None
    y_l_min: float | None = None
    y_l_max: float | None = None

    recovery_success: bool | None = None
    property_success: bool | None = None
    enthalpy_inversion_success: bool | None = None
    interface_domain_ok: bool | None = None

    nonfinite_state_detected: bool = False
    nonfinite_residual_detected: bool = False

    meta: dict[str, object] = field(default_factory=dict)


@dataclass(slots=True, kw_only=True)
class GuardCheckResult:
    """Structured result of one guard evaluation.

    Produced by the pure guard check functions and consumed by PETSc adapter
    wrappers, solver diagnostics, and step-level failure mapping.
    """

    ok: bool = True
    changed: bool = False
    guard_active: bool = False

    clip_ts: bool = False
    clip_mpp: bool = False
    shrink_suggested: bool = False

    failure_reason: GuardFailureReason = GuardFailureReason.NONE
    failure: FailureInfo = field(default_factory=FailureInfo)

    diagnostics: dict[str, object] = field(default_factory=dict)


def _get_nested(source: object, *path: str, default: object = None) -> object:
    current: object = source
    for key in path:
        if current is None:
            return default
        if isinstance(current, dict):
            if key not in current:
                return default
            current = current[key]
            continue
        if not hasattr(current, key):
            return default
        current = getattr(current, key)
    return current


def _to_float_or_none(value: object) -> float | None:
    if value is None:
        return None
    number = float(value)
    if not isfinite(number):
        raise ValueError("guard configuration numeric values must be finite when provided")
    return number


def _choose_explicit_or_default(explicit: float | None, default: float | None) -> float | None:
    """Return explicit value when provided; otherwise fall back to default."""

    return default if explicit is None else explicit


def _default_ts_bounds_from_ctx(ctx: NonlinearContext) -> tuple[float | None, float | None]:
    tmins = [
        _to_float_or_none(_get_nested(ctx.cfg, "recovery", "T_min_l")),
        _to_float_or_none(_get_nested(ctx.cfg, "recovery", "T_min_g")),
    ]
    tmaxs = [
        _to_float_or_none(_get_nested(ctx.cfg, "recovery", "T_max_l")),
        _to_float_or_none(_get_nested(ctx.cfg, "recovery", "T_max_g")),
    ]
    lower_candidates = [val for val in tmins if val is not None]
    upper_candidates = [val for val in tmaxs if val is not None]
    lower = min(lower_candidates) if lower_candidates else None
    upper = max(upper_candidates) if upper_candidates else None
    return lower, upper


def build_linesearch_guard_config(ctx: NonlinearContext) -> LineSearchGuardConfig:
    """Build guard configuration from solver/recovery config and context metadata."""

    guard_cfg = _get_nested(ctx.cfg, "solver_inner_petsc", "guard", default=None)
    if guard_cfg is None:
        guard_cfg = _get_nested(ctx.cfg, "solver_inner_petsc", "linesearch_guard", default=None)
    meta_guard = ctx.meta.get("guard") if isinstance(ctx.meta, dict) else None
    ts_default_min, ts_default_max = _default_ts_bounds_from_ctx(ctx)

    def _first(*paths: tuple[object, str]) -> object:
        for source, name in paths:
            if source is None:
                continue
            value = _get_nested(source, name, default=None)
            if value is not None:
                return value
        return None

    enabled = _first((meta_guard, "enabled"), (guard_cfg, "enabled"))
    enable_precheck = _first((meta_guard, "enable_precheck"), (guard_cfg, "enable_precheck"))
    enable_postcheck = _first((meta_guard, "enable_postcheck"), (guard_cfg, "enable_postcheck"))
    allow_small_ts_clip = _first((meta_guard, "allow_small_ts_clip"), (guard_cfg, "allow_small_ts_clip"))
    ts_clip_abs_tol = _first((meta_guard, "ts_clip_abs_tol"), (guard_cfg, "ts_clip_abs_tol"))
    ts_hard_min = _to_float_or_none(_first((meta_guard, "ts_hard_min"), (guard_cfg, "ts_hard_min")))
    ts_hard_max = _to_float_or_none(_first((meta_guard, "ts_hard_max"), (guard_cfg, "ts_hard_max")))
    allow_small_mpp_clip = _first((meta_guard, "allow_small_mpp_clip"), (guard_cfg, "allow_small_mpp_clip"))
    mpp_clip_abs_tol = _first((meta_guard, "mpp_clip_abs_tol"), (guard_cfg, "mpp_clip_abs_tol"))
    composition_hard_tol = _first((meta_guard, "composition_hard_tol"), (guard_cfg, "composition_hard_tol"))
    density_min = _first((meta_guard, "density_min"), (guard_cfg, "density_min"))
    require_finite_state = _first((meta_guard, "require_finite_state"), (guard_cfg, "require_finite_state"))
    require_finite_residual = _first((meta_guard, "require_finite_residual"), (guard_cfg, "require_finite_residual"))

    cfg = LineSearchGuardConfig(
        enabled=bool(True if enabled is None else enabled),
        enable_precheck=bool(True if enable_precheck is None else enable_precheck),
        enable_postcheck=bool(True if enable_postcheck is None else enable_postcheck),
        allow_small_ts_clip=bool(True if allow_small_ts_clip is None else allow_small_ts_clip),
        ts_clip_abs_tol=float(1.0e-8 if ts_clip_abs_tol is None else ts_clip_abs_tol),
        ts_hard_min=_choose_explicit_or_default(ts_hard_min, ts_default_min),
        ts_hard_max=_choose_explicit_or_default(ts_hard_max, ts_default_max),
        allow_small_mpp_clip=bool(False if allow_small_mpp_clip is None else allow_small_mpp_clip),
        mpp_clip_abs_tol=float(0.0 if mpp_clip_abs_tol is None else mpp_clip_abs_tol),
        mpp_hard_min=_to_float_or_none(_first((meta_guard, "mpp_hard_min"), (guard_cfg, "mpp_hard_min"))),
        mpp_hard_max=_to_float_or_none(_first((meta_guard, "mpp_hard_max"), (guard_cfg, "mpp_hard_max"))),
        temperature_hard_min=ts_default_min,
        temperature_hard_max=ts_default_max,
        composition_hard_tol=float(1.0e-12 if composition_hard_tol is None else composition_hard_tol),
        density_min=float(1.0e-300 if density_min is None else density_min),
        require_finite_state=bool(True if require_finite_state is None else require_finite_state),
        require_finite_residual=bool(True if require_finite_residual is None else require_finite_residual),
    )
    return cfg


def _resolve_scalar_index(layout: object, *, index_attr: str, slice_attr: str) -> int:
    """Resolve a scalar unknown location from layout via index or slice."""

    if hasattr(layout, index_attr):
        return int(getattr(layout, index_attr))
    if hasattr(layout, slice_attr):
        sl = getattr(layout, slice_attr)
        if sl.stop - sl.start <= 0:
            raise ValueError(f"{slice_attr} does not contain a scalar entry")
        return int(sl.start)
    raise ValueError(f"layout must provide either {index_attr} or {slice_attr}")


def extract_trial_scalars_from_array(*, u_trial_array: object, layout: object) -> dict[str, float | None]:
    """Extract cheap interface scalar unknowns from a trial vector using layout only."""

    arr = np.asarray(u_trial_array, dtype=np.float64).reshape(-1)
    total_size = getattr(layout, "total_size", None)
    if total_size is not None and arr.size != int(total_size):
        raise ValueError("trial vector length must match layout.total_size")

    Ts = float(arr[_resolve_scalar_index(layout, index_attr="if_temperature_index", slice_attr="if_temperature_slice")])
    mpp = float(arr[_resolve_scalar_index(layout, index_attr="if_mpp_index", slice_attr="if_mpp_slice")])

    return {"Ts": Ts, "mpp": mpp}


def make_guard_failure(
    *,
    reason: GuardFailureReason,
    where: str,
    message: str,
    recoverable: bool,
    rollback_required: bool = False,
    meta: dict[str, object] | None = None,
) -> FailureInfo:
    """Create a structured guard failure payload."""

    return FailureInfo(
        failure_class=FailureClass.GUARD_FAIL,
        reason_code=str(reason.value),
        message=message,
        where=where,
        recoverable=bool(recoverable),
        rollback_required=bool(rollback_required),
        meta=dict(meta) if meta is not None else {},
    )


def _success_result(*, diagnostics: dict[str, object] | None = None, changed: bool = False, guard_active: bool = False, clip_ts: bool = False, clip_mpp: bool = False) -> GuardCheckResult:
    return GuardCheckResult(
        ok=True,
        changed=changed,
        guard_active=guard_active,
        clip_ts=clip_ts,
        clip_mpp=clip_mpp,
        shrink_suggested=False,
        failure_reason=GuardFailureReason.NONE,
        failure=FailureInfo(),
        diagnostics=dict(diagnostics) if diagnostics is not None else {},
    )


def check_trial_vector(
    *,
    ctx: NonlinearContext,
    u_trial_array: object,
    cfg_guard: LineSearchGuardConfig,
) -> GuardCheckResult:
    """Run cheap legality checks on a trial vector before/around line search."""

    _ = ctx
    if not cfg_guard.enabled or not cfg_guard.enable_precheck:
        return _success_result()

    arr = np.asarray(u_trial_array)
    if arr.ndim != 1:
        arr = arr.reshape(-1)
    if not np.issubdtype(arr.dtype, np.floating):
        raise ValueError("trial vector view must use a floating dtype")
    diagnostics: dict[str, object] = {}
    if cfg_guard.require_finite_state and not np.all(np.isfinite(arr)):
        failure = make_guard_failure(
            reason=GuardFailureReason.NONFINITE_STATE,
            where="linesearch_precheck",
            message="trial vector contains non-finite values",
            recoverable=True,
            meta={"nonfinite_state_detected": True},
        )
        return GuardCheckResult(
            ok=False,
            changed=False,
            guard_active=True,
            failure_reason=GuardFailureReason.NONFINITE_STATE,
            failure=failure,
            diagnostics={"nonfinite_state_detected": True},
        )

    ts_index = _resolve_scalar_index(ctx.layout, index_attr="if_temperature_index", slice_attr="if_temperature_slice")
    mpp_index = _resolve_scalar_index(ctx.layout, index_attr="if_mpp_index", slice_attr="if_mpp_slice")
    scalars = extract_trial_scalars_from_array(u_trial_array=arr, layout=ctx.layout)
    diagnostics.update({"Ts": scalars["Ts"], "mpp": scalars["mpp"]})
    changed = False
    clip_ts = False
    clip_mpp = False

    ts = scalars["Ts"]
    if ts is not None:
        if cfg_guard.ts_hard_min is not None and ts < cfg_guard.ts_hard_min:
            delta = cfg_guard.ts_hard_min - ts
            if cfg_guard.allow_small_ts_clip and delta <= cfg_guard.ts_clip_abs_tol:
                if not arr.flags.writeable:
                    raise ValueError("trial vector view must be writable for in-place Ts clipping")
                arr[ts_index] = float(cfg_guard.ts_hard_min)
                changed = True
                clip_ts = True
                diagnostics["Ts_clipped_to"] = float(cfg_guard.ts_hard_min)
            else:
                failure = make_guard_failure(
                    reason=GuardFailureReason.TEMPERATURE_OUT_OF_RANGE,
                    where="linesearch_precheck",
                    message="interface temperature fell below hard minimum",
                    recoverable=True,
                    meta={"Ts": ts, "ts_hard_min": cfg_guard.ts_hard_min},
                )
                return GuardCheckResult(
                    ok=False,
                    changed=False,
                    guard_active=True,
                    failure_reason=GuardFailureReason.TEMPERATURE_OUT_OF_RANGE,
                    failure=failure,
                    diagnostics={"Ts": ts, "ts_hard_min": cfg_guard.ts_hard_min},
                )
        if cfg_guard.ts_hard_max is not None and ts > cfg_guard.ts_hard_max:
            delta = ts - cfg_guard.ts_hard_max
            if cfg_guard.allow_small_ts_clip and delta <= cfg_guard.ts_clip_abs_tol:
                if not arr.flags.writeable:
                    raise ValueError("trial vector view must be writable for in-place Ts clipping")
                arr[ts_index] = float(cfg_guard.ts_hard_max)
                changed = True
                clip_ts = True
                diagnostics["Ts_clipped_to"] = float(cfg_guard.ts_hard_max)
            else:
                failure = make_guard_failure(
                    reason=GuardFailureReason.TEMPERATURE_OUT_OF_RANGE,
                    where="linesearch_precheck",
                    message="interface temperature exceeded hard maximum",
                    recoverable=True,
                    meta={"Ts": ts, "ts_hard_max": cfg_guard.ts_hard_max},
                )
                return GuardCheckResult(
                    ok=False,
                    changed=False,
                    guard_active=True,
                    failure_reason=GuardFailureReason.TEMPERATURE_OUT_OF_RANGE,
                    failure=failure,
                    diagnostics={"Ts": ts, "ts_hard_max": cfg_guard.ts_hard_max},
                )

    mpp = scalars["mpp"]
    if mpp is not None:
        if cfg_guard.mpp_hard_min is not None and mpp < cfg_guard.mpp_hard_min:
            delta = cfg_guard.mpp_hard_min - mpp
            if cfg_guard.allow_small_mpp_clip and delta <= cfg_guard.mpp_clip_abs_tol:
                if not arr.flags.writeable:
                    raise ValueError("trial vector view must be writable for in-place mpp clipping")
                arr[mpp_index] = float(cfg_guard.mpp_hard_min)
                changed = True
                clip_mpp = True
                diagnostics["mpp_clipped_to"] = float(cfg_guard.mpp_hard_min)
            else:
                failure = make_guard_failure(
                    reason=GuardFailureReason.INTERFACE_DOMAIN_ERROR,
                    where="linesearch_precheck",
                    message="mpp fell below hard minimum",
                    recoverable=True,
                    meta={"mpp": mpp, "mpp_hard_min": cfg_guard.mpp_hard_min},
                )
                return GuardCheckResult(
                    ok=False,
                    changed=False,
                    guard_active=True,
                    failure_reason=GuardFailureReason.INTERFACE_DOMAIN_ERROR,
                    failure=failure,
                    diagnostics={"mpp": mpp, "mpp_hard_min": cfg_guard.mpp_hard_min},
                )
        if cfg_guard.mpp_hard_max is not None and mpp > cfg_guard.mpp_hard_max:
            delta = mpp - cfg_guard.mpp_hard_max
            if cfg_guard.allow_small_mpp_clip and delta <= cfg_guard.mpp_clip_abs_tol:
                if not arr.flags.writeable:
                    raise ValueError("trial vector view must be writable for in-place mpp clipping")
                arr[mpp_index] = float(cfg_guard.mpp_hard_max)
                changed = True
                clip_mpp = True
                diagnostics["mpp_clipped_to"] = float(cfg_guard.mpp_hard_max)
            else:
                failure = make_guard_failure(
                    reason=GuardFailureReason.INTERFACE_DOMAIN_ERROR,
                    where="linesearch_precheck",
                    message="mpp exceeded hard maximum",
                    recoverable=True,
                    meta={"mpp": mpp, "mpp_hard_max": cfg_guard.mpp_hard_max},
                )
                return GuardCheckResult(
                    ok=False,
                    changed=False,
                    guard_active=True,
                    failure_reason=GuardFailureReason.INTERFACE_DOMAIN_ERROR,
                    failure=failure,
                    diagnostics={"mpp": mpp, "mpp_hard_max": cfg_guard.mpp_hard_max},
                )

    diagnostics["guard_active"] = bool(changed)
    diagnostics["ls_clip_ts"] = bool(clip_ts)
    diagnostics["ls_clip_mpp"] = bool(clip_mpp)
    return _success_result(
        diagnostics=diagnostics,
        changed=changed,
        guard_active=changed,
        clip_ts=clip_ts,
        clip_mpp=clip_mpp,
    )


def check_trial_snapshot(
    *,
    ctx: NonlinearContext,
    snapshot: TrialPhysicsSnapshot,
    cfg_guard: LineSearchGuardConfig,
) -> GuardCheckResult:
    """Run post-evaluation guard checks on a trial physics snapshot."""

    _ = ctx
    if not cfg_guard.enabled or not cfg_guard.enable_postcheck:
        return _success_result()

    if snapshot.nonfinite_state_detected:
        failure = make_guard_failure(
            reason=GuardFailureReason.NONFINITE_STATE,
            where="linesearch_postcheck",
            message="trial physics snapshot detected non-finite state values",
            recoverable=True,
            meta={"nonfinite_state_detected": True},
        )
        return GuardCheckResult(
            ok=False,
            guard_active=True,
            failure_reason=GuardFailureReason.NONFINITE_STATE,
            failure=failure,
            diagnostics={"nonfinite_state_detected": True},
        )

    if cfg_guard.require_finite_residual and snapshot.nonfinite_residual_detected:
        failure = make_guard_failure(
            reason=GuardFailureReason.NONFINITE_RESIDUAL,
            where="linesearch_postcheck",
            message="trial evaluation produced a non-finite residual",
            recoverable=True,
            meta={"nonfinite_residual_detected": True},
        )
        return GuardCheckResult(
            ok=False,
            guard_active=True,
            failure_reason=GuardFailureReason.NONFINITE_RESIDUAL,
            failure=failure,
            diagnostics={"nonfinite_residual_detected": True},
        )

    for temp_name, value in (
        ("Tl_min", snapshot.Tl_min),
        ("Tl_max", snapshot.Tl_max),
        ("Tg_min", snapshot.Tg_min),
        ("Tg_max", snapshot.Tg_max),
        ("Ts", snapshot.Ts),
    ):
        if value is None:
            continue
        if cfg_guard.temperature_hard_min is not None and temp_name.endswith("min") and value < cfg_guard.temperature_hard_min:
            failure = make_guard_failure(
                reason=GuardFailureReason.TEMPERATURE_OUT_OF_RANGE,
                where="linesearch_postcheck",
                message=f"{temp_name} fell below hard minimum",
                recoverable=True,
                meta={temp_name: value, "temperature_hard_min": cfg_guard.temperature_hard_min},
            )
            return GuardCheckResult(
                ok=False,
                guard_active=True,
                failure_reason=GuardFailureReason.TEMPERATURE_OUT_OF_RANGE,
                failure=failure,
                diagnostics={temp_name: value},
            )
        if cfg_guard.temperature_hard_max is not None and temp_name.endswith("max") and value > cfg_guard.temperature_hard_max:
            failure = make_guard_failure(
                reason=GuardFailureReason.TEMPERATURE_OUT_OF_RANGE,
                where="linesearch_postcheck",
                message=f"{temp_name} exceeded hard maximum",
                recoverable=True,
                meta={temp_name: value, "temperature_hard_max": cfg_guard.temperature_hard_max},
            )
            return GuardCheckResult(
                ok=False,
                guard_active=True,
                failure_reason=GuardFailureReason.TEMPERATURE_OUT_OF_RANGE,
                failure=failure,
                diagnostics={temp_name: value},
            )
    if snapshot.Ts is not None:
        if cfg_guard.ts_hard_min is not None and snapshot.Ts < cfg_guard.ts_hard_min:
            failure = make_guard_failure(
                reason=GuardFailureReason.TEMPERATURE_OUT_OF_RANGE,
                where="linesearch_postcheck",
                message="Ts fell below hard minimum",
                recoverable=True,
                meta={"Ts": snapshot.Ts, "ts_hard_min": cfg_guard.ts_hard_min},
            )
            return GuardCheckResult(
                ok=False,
                guard_active=True,
                failure_reason=GuardFailureReason.TEMPERATURE_OUT_OF_RANGE,
                failure=failure,
                diagnostics={"Ts": snapshot.Ts},
            )
        if cfg_guard.ts_hard_max is not None and snapshot.Ts > cfg_guard.ts_hard_max:
            failure = make_guard_failure(
                reason=GuardFailureReason.TEMPERATURE_OUT_OF_RANGE,
                where="linesearch_postcheck",
                message="Ts exceeded hard maximum",
                recoverable=True,
                meta={"Ts": snapshot.Ts, "ts_hard_max": cfg_guard.ts_hard_max},
            )
            return GuardCheckResult(
                ok=False,
                guard_active=True,
                failure_reason=GuardFailureReason.TEMPERATURE_OUT_OF_RANGE,
                failure=failure,
                diagnostics={"Ts": snapshot.Ts},
            )

    tol = float(cfg_guard.composition_hard_tol)
    for name, value, low, high in (
        ("y_g_min", snapshot.y_g_min, True, False),
        ("y_l_min", snapshot.y_l_min, True, False),
        ("y_g_max", snapshot.y_g_max, False, True),
        ("y_l_max", snapshot.y_l_max, False, True),
    ):
        if value is None:
            continue
        if low and value < -tol:
            failure = make_guard_failure(
                reason=GuardFailureReason.COMPOSITION_OUT_OF_RANGE,
                where="linesearch_postcheck",
                message=f"{name} fell below admissible lower bound",
                recoverable=True,
                meta={name: value, "composition_hard_tol": tol},
            )
            return GuardCheckResult(
                ok=False,
                guard_active=True,
                failure_reason=GuardFailureReason.COMPOSITION_OUT_OF_RANGE,
                failure=failure,
                diagnostics={name: value},
            )
        if high and value > 1.0 + tol:
            failure = make_guard_failure(
                reason=GuardFailureReason.COMPOSITION_OUT_OF_RANGE,
                where="linesearch_postcheck",
                message=f"{name} exceeded admissible upper bound",
                recoverable=True,
                meta={name: value, "composition_hard_tol": tol},
            )
            return GuardCheckResult(
                ok=False,
                guard_active=True,
                failure_reason=GuardFailureReason.COMPOSITION_OUT_OF_RANGE,
                failure=failure,
                diagnostics={name: value},
            )

    if snapshot.rho_l_min is not None and snapshot.rho_l_min <= cfg_guard.density_min:
        failure = make_guard_failure(
            reason=GuardFailureReason.NEGATIVE_DENSITY,
            where="linesearch_postcheck",
            message="rho_l_min fell below admissible density floor",
            recoverable=True,
            meta={"rho_l_min": snapshot.rho_l_min, "density_min": cfg_guard.density_min},
        )
        return GuardCheckResult(
            ok=False,
            guard_active=True,
            failure_reason=GuardFailureReason.NEGATIVE_DENSITY,
            failure=failure,
            diagnostics={"rho_l_min": snapshot.rho_l_min},
        )
    if snapshot.rho_g_min is not None and snapshot.rho_g_min <= cfg_guard.density_min:
        failure = make_guard_failure(
            reason=GuardFailureReason.NEGATIVE_DENSITY,
            where="linesearch_postcheck",
            message="rho_g_min fell below admissible density floor",
            recoverable=True,
            meta={"rho_g_min": snapshot.rho_g_min, "density_min": cfg_guard.density_min},
        )
        return GuardCheckResult(
            ok=False,
            guard_active=True,
            failure_reason=GuardFailureReason.NEGATIVE_DENSITY,
            failure=failure,
            diagnostics={"rho_g_min": snapshot.rho_g_min},
        )

    if snapshot.property_success is False:
        failure = make_guard_failure(
            reason=GuardFailureReason.PROPERTY_EVAL_FAILED,
            where="linesearch_postcheck",
            message="trial evaluation reported property failure",
            recoverable=True,
            meta={"property_success": False},
        )
        return GuardCheckResult(
            ok=False,
            guard_active=True,
            failure_reason=GuardFailureReason.PROPERTY_EVAL_FAILED,
            failure=failure,
            diagnostics={"property_success": False},
        )

    if snapshot.enthalpy_inversion_success is False or snapshot.recovery_success is False:
        failure = make_guard_failure(
            reason=GuardFailureReason.ENTHALPY_INVERSION_FAILED,
            where="linesearch_postcheck",
            message="trial evaluation reported enthalpy inversion / recovery failure",
            recoverable=True,
            meta={
                "enthalpy_inversion_success": snapshot.enthalpy_inversion_success,
                "recovery_success": snapshot.recovery_success,
            },
        )
        return GuardCheckResult(
            ok=False,
            guard_active=True,
            failure_reason=GuardFailureReason.ENTHALPY_INVERSION_FAILED,
            failure=failure,
            diagnostics={
                "enthalpy_inversion_success": snapshot.enthalpy_inversion_success,
                "recovery_success": snapshot.recovery_success,
            },
        )

    if snapshot.interface_domain_ok is False:
        failure = make_guard_failure(
            reason=GuardFailureReason.INTERFACE_DOMAIN_ERROR,
            where="linesearch_postcheck",
            message="trial evaluation reported interface domain failure",
            recoverable=True,
            meta={"interface_domain_ok": False},
        )
        return GuardCheckResult(
            ok=False,
            guard_active=True,
            failure_reason=GuardFailureReason.INTERFACE_DOMAIN_ERROR,
            failure=failure,
            diagnostics={"interface_domain_ok": False},
        )

    return _success_result(
        diagnostics={
            "guard_active": False,
            "nonfinite_state_detected": bool(snapshot.nonfinite_state_detected),
            "nonfinite_residual_detected": bool(snapshot.nonfinite_residual_detected),
        }
    )


def update_guard_diagnostics(
    *,
    result: GuardCheckResult,
    diagnostics: dict[str, object] | None = None,
    ls_shrink_count: int | None = None,
    line_search_backtracks: int | None = None,
) -> dict[str, object]:
    """Merge structured guard outputs into a diagnostics dictionary."""

    target = dict(diagnostics) if diagnostics is not None else {}
    target.update(result.diagnostics)
    target["guard_active"] = bool(result.guard_active)
    target["guard_reason"] = result.failure_reason.value
    target["ls_clip_ts"] = bool(result.clip_ts)
    target["ls_clip_mpp"] = bool(result.clip_mpp)
    target["nonfinite_state_detected"] = bool(
        result.diagnostics.get("nonfinite_state_detected", False)
    )
    target["nonfinite_residual_detected"] = bool(
        result.diagnostics.get("nonfinite_residual_detected", False)
    )
    if ls_shrink_count is not None:
        target["ls_shrink_count"] = int(ls_shrink_count)
    else:
        target.setdefault("ls_shrink_count", 0)
    if line_search_backtracks is not None:
        target["line_search_backtracks"] = int(line_search_backtracks)
    else:
        target.setdefault("line_search_backtracks", 0)
    return target


def make_petsc_linesearch_precheck(
    *,
    ctx: NonlinearContext,
    cfg_guard: LineSearchGuardConfig,
    array_view_factory: Any,
    on_guard_result: Any | None = None,
    on_domain_error: Any | None = None,
) -> Any:
    """Create a PETSc-style precheck adapter backed by ``check_trial_vector``."""

    def _precheck(_linesearch: object, _x: object, y: object, *args: object) -> bool:
        _ = args
        trial_array = array_view_factory(y, writable=True)
        result = check_trial_vector(ctx=ctx, u_trial_array=trial_array, cfg_guard=cfg_guard)
        ctx.diagnostics = update_guard_diagnostics(result=result, diagnostics=ctx.diagnostics)
        ctx.meta["last_guard_result"] = result
        if on_guard_result is not None:
            on_guard_result(result)
        if not result.ok and on_domain_error is not None:
            on_domain_error(result)
        return bool(result.changed)

    return _precheck


def make_petsc_linesearch_postcheck(
    *,
    ctx: NonlinearContext,
    cfg_guard: LineSearchGuardConfig,
    snapshot_getter: Any,
    on_guard_result: Any | None = None,
    on_domain_error: Any | None = None,
) -> Any:
    """Create a PETSc-style postcheck adapter backed by ``check_trial_snapshot``."""

    def _postcheck(_linesearch: object, _x: object, _y: object, _w: object | None = None, *args: object) -> bool:
        _ = args
        snapshot = snapshot_getter()
        if not isinstance(snapshot, TrialPhysicsSnapshot):
            raise ValueError("snapshot_getter must return a TrialPhysicsSnapshot")
        result = check_trial_snapshot(ctx=ctx, snapshot=snapshot, cfg_guard=cfg_guard)
        ctx.diagnostics = update_guard_diagnostics(result=result, diagnostics=ctx.diagnostics)
        ctx.meta["last_guard_result"] = result
        if on_guard_result is not None:
            on_guard_result(result)
        if not result.ok and on_domain_error is not None:
            on_domain_error(result)
        return bool(result.changed)

    return _postcheck


__all__ = [
    "GuardCheckResult",
    "LineSearchGuardConfig",
    "TrialPhysicsSnapshot",
    "build_linesearch_guard_config",
    "check_trial_snapshot",
    "check_trial_vector",
    "extract_trial_scalars_from_array",
    "make_guard_failure",
    "make_petsc_linesearch_postcheck",
    "make_petsc_linesearch_precheck",
    "update_guard_diagnostics",
]
