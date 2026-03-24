from __future__ import annotations

"""Step-attempt acceptance decision logic for Phase 5.

This module only evaluates the terminal state of one step attempt and returns
an acceptance / retry / fatal decision. It does not continue outer iterations,
run predictor/corrector/inner solves, execute rollback, or orchestrate the
time-step loop itself.
"""

from dataclasses import dataclass
from math import isfinite

from .nonlinear_types import FailureClass, FailureInfo, StepAcceptanceDecision, StepAction


@dataclass(slots=True, kw_only=True)
class StepAcceptanceConfigView:
    """Normalized step-acceptance and retry policy configuration."""

    dt_min: float
    dt_max: float
    reject_shrink_factor: float = 0.5
    accept_growth_factor: float = 1.0
    max_retries_per_step: int = 8


def _validate_positive_finite(name: str, value: float) -> float:
    """Validate one positive finite scalar configuration/input value."""

    value_f = float(value)
    if not isfinite(value_f):
        raise ValueError(f"{name} must be finite")
    if value_f <= 0.0:
        raise ValueError(f"{name} must be > 0")
    return value_f


def _validate_nonnegative_int(name: str, value: int) -> int:
    """Validate one non-negative integer configuration/input value."""

    ivalue = int(value)
    if ivalue < 0:
        raise ValueError(f"{name} must be >= 0")
    return ivalue


def build_step_acceptance_config_view(cfg: object) -> StepAcceptanceConfigView:
    """Build a normalized step-acceptance config view from runtime config."""

    ts = getattr(cfg, "time_stepper", None)

    dt_min = _validate_positive_finite("dt_min", float(getattr(ts, "dt_min", 1.0e-9)))
    dt_max = _validate_positive_finite("dt_max", float(getattr(ts, "dt_max", 1.0e9)))
    if dt_max < dt_min:
        raise ValueError("dt_max must be >= dt_min")

    reject_shrink = _validate_positive_finite(
        "reject_shrink_factor",
        float(getattr(ts, "reject_shrink_factor", 0.5)),
    )
    if reject_shrink >= 1.0:
        raise ValueError("reject_shrink_factor must be < 1")

    accept_growth = _validate_positive_finite(
        "accept_growth_factor",
        float(getattr(ts, "accept_growth_factor", 1.0)),
    )
    if accept_growth < 1.0:
        raise ValueError("accept_growth_factor must be >= 1")

    max_retries = _validate_nonnegative_int(
        "max_retries_per_step",
        int(getattr(ts, "max_retries_per_step", 8)),
    )

    return StepAcceptanceConfigView(
        dt_min=dt_min,
        dt_max=dt_max,
        reject_shrink_factor=reject_shrink,
        accept_growth_factor=accept_growth,
        max_retries_per_step=max_retries,
    )


def compute_dt_next_on_accept(
    *,
    dt_current: float,
    dt_max: float,
    accept_growth_factor: float,
) -> float:
    """Compute the next candidate dt after an accepted step."""

    return min(float(dt_current) * float(accept_growth_factor), float(dt_max))


def compute_dt_next_on_reject(
    *,
    dt_current: float,
    dt_min: float,
    reject_shrink_factor: float,
) -> float:
    """Compute the next candidate dt after a rejected step."""

    return max(float(dt_current) * float(reject_shrink_factor), float(dt_min))


def build_failure_for_inner_not_converged(inner_result: object) -> FailureInfo:
    """Build a structured inner-failure payload for step rejection."""

    failure = getattr(inner_result, "failure", None)
    if isinstance(failure, FailureInfo) and failure.failure_class is not FailureClass.NONE:
        return FailureInfo(
            failure_class=failure.failure_class,
            reason_code=failure.reason_code,
            message=failure.message,
            where=failure.where,
            recoverable=failure.recoverable,
            rollback_required=True,
            meta=dict(failure.meta),
        )
    return FailureInfo(
        failure_class=FailureClass.INNER_FAIL,
        reason_code="inner_not_converged",
        message="inner solve did not converge",
        where="step_acceptance",
        recoverable=True,
        rollback_required=True,
    )


def build_failure_for_outer_not_converged(*, outer_iter_count: int, outer_iter_limit: int) -> FailureInfo:
    """Build a structured outer-failure payload for step rejection."""

    return FailureInfo(
        failure_class=FailureClass.OUTER_FAIL,
        reason_code="outer_max_iter_exhausted",
        message="outer iteration limit exhausted without convergence",
        where="step_acceptance",
        recoverable=True,
        rollback_required=True,
        meta={
            "outer_iter_count": int(outer_iter_count),
            "outer_iter_limit": int(outer_iter_limit),
        },
    )


def build_failure_for_fatal_stop(
    *,
    reason_code: str,
    message: str,
    cause: FailureInfo | None = None,
) -> FailureInfo:
    """Build a structured fatal-stop payload."""

    meta: dict[str, object] = {}
    if isinstance(cause, FailureInfo):
        meta["cause_failure_class"] = cause.failure_class.value
        meta["cause_reason_code"] = cause.reason_code
        if cause.meta:
            meta["cause_meta"] = dict(cause.meta)
    return FailureInfo(
        failure_class=FailureClass.FATAL_STOP,
        reason_code=reason_code,
        message=message,
        where="step_acceptance",
        recoverable=False,
        rollback_required=True,
        meta=meta,
    )


def _coerce_fatal_failure(fatal_failure: object) -> FailureInfo:
    """Coerce an upstream fatal failure object into a fatal-stop payload."""

    if isinstance(fatal_failure, FailureInfo):
        if fatal_failure.failure_class is FailureClass.FATAL_STOP and not fatal_failure.recoverable:
            return FailureInfo(
                failure_class=FailureClass.FATAL_STOP,
                reason_code=fatal_failure.reason_code,
                message=fatal_failure.message,
                where=fatal_failure.where or "step_acceptance",
                recoverable=False,
                rollback_required=True,
                meta=dict(fatal_failure.meta),
            )
        return build_failure_for_fatal_stop(
            reason_code=fatal_failure.reason_code or "fatal_failure",
            message=fatal_failure.message or "fatal failure propagated to step acceptance",
            cause=fatal_failure,
        )
    return build_failure_for_fatal_stop(
        reason_code="fatal_failure",
        message="fatal failure propagated to step acceptance",
    )


def _common_diagnostics(
    *,
    dt_current: float,
    dt_next: float,
    retries_used: int,
    outer_iter_count: int,
    outer_iter_limit: int,
    inner_converged: bool,
    outer_converged: bool | None,
) -> dict[str, object]:
    """Build common diagnostics shared by all decision branches."""

    return {
        "dt_current": float(dt_current),
        "dt_next": float(dt_next),
        "retries_used": int(retries_used),
        "outer_iter_count": int(outer_iter_count),
        "outer_iter_limit": int(outer_iter_limit),
        "inner_converged": bool(inner_converged),
        "outer_converged": None if outer_converged is None else bool(outer_converged),
    }


def decide_step_acceptance(
    *,
    cfg: object,
    dt_current: float,
    retries_used: int,
    inner_result: object | None,
    outer_convergence: object | None,
    outer_iter_count: int,
    outer_iter_limit: int,
    accepted_state_candidate: object | None = None,
    accepted_geometry_candidate: object | None = None,
    fatal_failure: object | None = None,
) -> StepAcceptanceDecision:
    """Return the structured acceptance / retry / fatal decision for one step attempt."""

    cfg_acc = build_step_acceptance_config_view(cfg)

    dt_current_f = _validate_positive_finite("dt_current", float(dt_current))
    retries_used_i = _validate_nonnegative_int("retries_used", int(retries_used))
    outer_iter_count_i = _validate_nonnegative_int("outer_iter_count", int(outer_iter_count))
    outer_iter_limit_i = _validate_nonnegative_int("outer_iter_limit", int(outer_iter_limit))

    if fatal_failure is not None:
        failure = _coerce_fatal_failure(fatal_failure)
        decision = StepAcceptanceDecision(
            action=StepAction.FATAL_STOP,
            accepted=False,
            rollback_required=True,
            dt_current=dt_current_f,
            dt_next=dt_current_f,
            reject_reason=failure.reason_code or "fatal_stop",
            failure=failure,
            diagnostics=_common_diagnostics(
                dt_current=dt_current_f,
                dt_next=dt_current_f,
                retries_used=retries_used_i,
                outer_iter_count=outer_iter_count_i,
                outer_iter_limit=outer_iter_limit_i,
                inner_converged=False,
                outer_converged=None,
            ),
        )
        decision.assert_consistent()
        return decision

    if inner_result is None:
        raise ValueError("step_acceptance requires inner_result")

    inner_converged = bool(getattr(inner_result, "converged", False))
    if not inner_converged:
        failure = build_failure_for_inner_not_converged(inner_result)
        if failure.failure_class is FailureClass.FATAL_STOP or not failure.recoverable:
            fatal = build_failure_for_fatal_stop(
                reason_code=failure.reason_code or "inner_fatal_failure",
                message=failure.message or "inner failure is not recoverable",
                cause=failure,
            )
            decision = StepAcceptanceDecision(
                action=StepAction.FATAL_STOP,
                accepted=False,
                rollback_required=True,
                dt_current=dt_current_f,
                dt_next=dt_current_f,
                reject_reason=fatal.reason_code,
                failure=fatal,
                diagnostics=_common_diagnostics(
                    dt_current=dt_current_f,
                    dt_next=dt_current_f,
                    retries_used=retries_used_i,
                    outer_iter_count=outer_iter_count_i,
                    outer_iter_limit=outer_iter_limit_i,
                    inner_converged=False,
                    outer_converged=None,
                ),
            )
            decision.assert_consistent()
            return decision
        can_retry = dt_current_f > cfg_acc.dt_min and retries_used_i < cfg_acc.max_retries_per_step
        if can_retry:
            dt_next = compute_dt_next_on_reject(
                dt_current=dt_current_f,
                dt_min=cfg_acc.dt_min,
                reject_shrink_factor=cfg_acc.reject_shrink_factor,
            )
            decision = StepAcceptanceDecision(
                action=StepAction.RETRY_REDUCED_DT,
                accepted=False,
                rollback_required=True,
                dt_current=dt_current_f,
                dt_next=dt_next,
                reject_reason=failure.reason_code or "inner_not_converged",
                failure=failure,
                diagnostics=_common_diagnostics(
                    dt_current=dt_current_f,
                    dt_next=dt_next,
                    retries_used=retries_used_i,
                    outer_iter_count=outer_iter_count_i,
                    outer_iter_limit=outer_iter_limit_i,
                    inner_converged=False,
                    outer_converged=None,
                ),
            )
            decision.assert_consistent()
            return decision
        fatal = build_failure_for_fatal_stop(
            reason_code="inner_retry_budget_exhausted"
            if retries_used_i >= cfg_acc.max_retries_per_step
            else "dt_min_reached_after_inner_fail",
            message="inner solve failed and step cannot be retried",
            cause=failure,
        )
        decision = StepAcceptanceDecision(
            action=StepAction.FATAL_STOP,
            accepted=False,
            rollback_required=True,
            dt_current=dt_current_f,
            dt_next=dt_current_f,
            reject_reason=fatal.reason_code,
            failure=fatal,
            diagnostics=_common_diagnostics(
                dt_current=dt_current_f,
                dt_next=dt_current_f,
                retries_used=retries_used_i,
                outer_iter_count=outer_iter_count_i,
                outer_iter_limit=outer_iter_limit_i,
                inner_converged=False,
                outer_converged=None,
            ),
        )
        decision.assert_consistent()
        return decision

    if outer_convergence is None:
        raise ValueError("step_acceptance requires outer_convergence after inner convergence")

    outer_converged = bool(getattr(outer_convergence, "converged", False))
    if not outer_converged:
        if outer_iter_count_i < outer_iter_limit_i:
            raise ValueError("step_acceptance called before outer termination")
        failure = build_failure_for_outer_not_converged(
            outer_iter_count=outer_iter_count_i,
            outer_iter_limit=outer_iter_limit_i,
        )
        can_retry = dt_current_f > cfg_acc.dt_min and retries_used_i < cfg_acc.max_retries_per_step
        if can_retry:
            dt_next = compute_dt_next_on_reject(
                dt_current=dt_current_f,
                dt_min=cfg_acc.dt_min,
                reject_shrink_factor=cfg_acc.reject_shrink_factor,
            )
            decision = StepAcceptanceDecision(
                action=StepAction.RETRY_REDUCED_DT,
                accepted=False,
                rollback_required=True,
                dt_current=dt_current_f,
                dt_next=dt_next,
                reject_reason=failure.reason_code,
                failure=failure,
                diagnostics=_common_diagnostics(
                    dt_current=dt_current_f,
                    dt_next=dt_next,
                    retries_used=retries_used_i,
                    outer_iter_count=outer_iter_count_i,
                    outer_iter_limit=outer_iter_limit_i,
                    inner_converged=True,
                    outer_converged=False,
                ),
            )
            decision.assert_consistent()
            return decision
        fatal = build_failure_for_fatal_stop(
            reason_code="outer_retry_budget_exhausted"
            if retries_used_i >= cfg_acc.max_retries_per_step
            else "dt_min_reached_after_outer_fail",
            message="outer iteration failed and step cannot be retried",
            cause=failure,
        )
        decision = StepAcceptanceDecision(
            action=StepAction.FATAL_STOP,
            accepted=False,
            rollback_required=True,
            dt_current=dt_current_f,
            dt_next=dt_current_f,
            reject_reason=fatal.reason_code,
            failure=fatal,
            diagnostics=_common_diagnostics(
                dt_current=dt_current_f,
                dt_next=dt_current_f,
                retries_used=retries_used_i,
                outer_iter_count=outer_iter_count_i,
                outer_iter_limit=outer_iter_limit_i,
                inner_converged=True,
                outer_converged=False,
            ),
        )
        decision.assert_consistent()
        return decision

    if accepted_state_candidate is None or accepted_geometry_candidate is None:
        raise ValueError("accepted step requires accepted_state_candidate and accepted_geometry_candidate")
    if getattr(inner_result, "state_vec", None) is None:
        raise ValueError("accepted step requires inner_result.state_vec")

    dt_next = compute_dt_next_on_accept(
        dt_current=dt_current_f,
        dt_max=cfg_acc.dt_max,
        accept_growth_factor=cfg_acc.accept_growth_factor,
    )
    decision = StepAcceptanceDecision(
        action=StepAction.ACCEPT,
        accepted=True,
        rollback_required=False,
        dt_current=dt_current_f,
        dt_next=dt_next,
        reject_reason="",
        failure=FailureInfo(
            failure_class=FailureClass.NONE,
            reason_code="",
            message="",
            where="step_acceptance",
            recoverable=False,
            rollback_required=False,
        ),
        diagnostics=_common_diagnostics(
            dt_current=dt_current_f,
            dt_next=dt_next,
            retries_used=retries_used_i,
            outer_iter_count=outer_iter_count_i,
            outer_iter_limit=outer_iter_limit_i,
            inner_converged=True,
            outer_converged=True,
        ),
    )
    decision.assert_consistent()
    return decision


__all__ = [
    "StepAcceptanceConfigView",
    "_validate_nonnegative_int",
    "_validate_positive_finite",
    "build_failure_for_fatal_stop",
    "build_failure_for_inner_not_converged",
    "build_failure_for_outer_not_converged",
    "build_step_acceptance_config_view",
    "compute_dt_next_on_accept",
    "compute_dt_next_on_reject",
    "decide_step_acceptance",
]
