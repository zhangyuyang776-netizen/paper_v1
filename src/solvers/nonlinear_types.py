from __future__ import annotations

"""Structured nonlinear-solver result types shared across Phase 5 modules.

This module defines the shared result language for:
- fixed-geometry inner nonlinear solves
- outer predictor / corrector / convergence stages
- step-level acceptance / retry / failure handling

It is intentionally type-only: no solver logic, no PETSc object creation, no
assembly imports, and no numpy-based global state truth source.
"""

from dataclasses import dataclass, field
from enum import Enum


class SolverBackend(str, Enum):
    """Backend identifier reported by solver-side statistics objects.

    Produced primarily by ``petsc_snes.py`` and consumed by diagnostics,
    step-forensics, and tests.
    """

    PETSC_SNES = "petsc_snes"
    TEST_DUMMY = "test_dummy"


class FailureClass(str, Enum):
    """Project-level failure taxonomy for inner / outer / step orchestration.

    Produced by solver, recovery, and acceptance modules; consumed by
    ``step_acceptance.py`` and ``timestepper.py``.
    """

    NONE = "none"
    INNER_FAIL = "inner_fail"
    OUTER_FAIL = "outer_fail"
    REMAP_FAIL = "remap_fail"
    RECOVERY_FAIL = "recovery_fail"
    PROPERTY_FAIL = "property_fail"
    GUARD_FAIL = "guard_fail"
    ACCEPTANCE_FAIL = "acceptance_fail"
    FATAL_STOP = "fatal_stop"


class StepAction(str, Enum):
    """Step-level control action selected after an attempt is evaluated.

    Produced by ``step_acceptance.py`` and consumed by ``timestepper.py``.
    """

    ACCEPT = "accept"
    REJECT = "reject"
    RETRY_SAME_DT = "retry_same_dt"
    RETRY_REDUCED_DT = "retry_reduced_dt"
    FATAL_STOP = "fatal_stop"


class GuardFailureReason(str, Enum):
    """Reason codes reserved for linesearch / state-domain guard failures.

    Produced by future guard modules and propagated through ``FailureInfo``.
    """

    NONE = "none"
    NONFINITE_STATE = "nonfinite_state"
    NONFINITE_RESIDUAL = "nonfinite_residual"
    TEMPERATURE_OUT_OF_RANGE = "temperature_out_of_range"
    COMPOSITION_OUT_OF_RANGE = "composition_out_of_range"
    NEGATIVE_DENSITY = "negative_density"
    ENTHALPY_INVERSION_FAILED = "enthalpy_inversion_failed"
    PROPERTY_EVAL_FAILED = "property_eval_failed"
    INTERFACE_DOMAIN_ERROR = "interface_domain_error"
    UNKNOWN = "unknown"


class InnerEntrySource(str, Enum):
    """Source of the current inner entry state for one outer iteration."""

    ACCEPTED_TIME_LEVEL = "accepted_time_level"
    TRANSFER_FROM_PREVIOUS_OUTER = "transfer_from_previous_outer"


@dataclass(slots=True, kw_only=True)
class ResidualNorms:
    """Residual norm bundle shared by inner, outer, and acceptance diagnostics.

    Produced by solver/convergence modules and consumed by higher-level
    diagnostics and time-step control.
    """

    l2: float | None = None
    linf: float | None = None
    scaled_l2: float | None = None
    scaled_linf: float | None = None


@dataclass(slots=True, kw_only=True)
class FailureInfo:
    """Structured failure payload propagated across solver layers.

    Produced by solver, recovery, guard, and acceptance modules; consumed by
    outer orchestration and step-level retry/rollback policy.
    """

    failure_class: FailureClass = FailureClass.NONE
    reason_code: str = ""
    message: str = ""
    where: str = ""
    recoverable: bool = False
    rollback_required: bool = False
    meta: dict[str, object] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        """Return ``True`` when no failure has been recorded."""

        return self.failure_class is FailureClass.NONE


@dataclass(slots=True, kw_only=True)
class InnerSolveStats:
    """Detailed statistics for one fixed-geometry inner nonlinear solve.

    Produced by ``petsc_snes.py`` and consumed by outer-iteration logic,
    acceptance diagnostics, and timestep forensics.
    """

    backend: SolverBackend = SolverBackend.PETSC_SNES
    converged: bool = False
    snes_reason: int | None = None
    ksp_reason: int | None = None
    inner_iter_count: int = 0
    linear_iter_count: int = 0
    residual_norms: ResidualNorms = field(default_factory=ResidualNorms)
    history_linf: list[float] = field(default_factory=list)
    line_search_used: bool = False
    damping_used: bool = False
    guard_triggered: bool = False
    wall_time_s: float | None = None
    message: str = ""
    meta: dict[str, object] = field(default_factory=dict)


@dataclass(slots=True, kw_only=True, init=False)
class InnerSolveResult:
    """Result of one fixed-geometry inner solve attempt.

    Produced by ``petsc_snes.py`` and consumed by outer corrector,
    outer-convergence logic, and step acceptance.

    ``solution_vec`` is the formal backend-native converged state handle.
    ``state_vec`` is retained as a read-only compatibility alias during the
    transition away from the superseded contract.
    """

    converged: bool = False
    solution_vec: object | None = None
    dot_a_phys: float | None = None
    entry_source: InnerEntrySource | None = None
    stats: InnerSolveStats = field(default_factory=InnerSolveStats)
    failure: FailureInfo = field(default_factory=FailureInfo)
    diagnostics: dict[str, object] = field(default_factory=dict)

    def __init__(
        self,
        *,
        converged: bool = False,
        solution_vec: object | None = None,
        state_vec: object | None = None,
        dot_a_phys: float | None = None,
        entry_source: InnerEntrySource | str | None = None,
        stats: InnerSolveStats | None = None,
        failure: FailureInfo | None = None,
        diagnostics: dict[str, object] | None = None,
        old_state_on_current_geometry: object | None = None,
    ) -> None:
        if solution_vec is not None and state_vec is not None:
            raise ValueError("Provide at most one of solution_vec or state_vec")
        self.converged = bool(converged)
        self.solution_vec = solution_vec if solution_vec is not None else state_vec
        self.dot_a_phys = dot_a_phys
        self.entry_source = (
            None if entry_source is None else InnerEntrySource(entry_source)
        )
        self.stats = stats if stats is not None else InnerSolveStats()
        self.failure = failure if failure is not None else FailureInfo()
        self.diagnostics = dict(diagnostics) if diagnostics is not None else {}
        if old_state_on_current_geometry is not None:
            self.diagnostics.setdefault("legacy_old_state_on_current_geometry", old_state_on_current_geometry)

    @property
    def has_state(self) -> bool:
        """Return ``True`` when a backend-native state handle is present."""

        return self.solution_vec is not None

    @property
    def state_vec(self) -> object | None:
        """Compatibility alias for the formal ``solution_vec`` field."""

        return self.solution_vec

    @property
    def old_state_on_current_geometry(self) -> object | None:
        """Compatibility alias for legacy diagnostics-only access."""

        return self.diagnostics.get("legacy_old_state_on_current_geometry")

    def assert_consistent(self) -> None:
        """Raise ``ValueError`` when core inner-result invariants are violated."""

        if self.converged != self.stats.converged:
            raise ValueError("InnerSolveResult.converged must match stats.converged")
        if self.converged and self.failure.failure_class is not FailureClass.NONE:
            raise ValueError("converged inner solve must not carry a non-NONE failure_class")
        if self.converged and self.solution_vec is None:
            raise ValueError("converged inner solve must provide solution_vec")
        if self.entry_source is not None and not isinstance(self.entry_source, InnerEntrySource):
            raise ValueError("entry_source must be an InnerEntrySource when provided")


@dataclass(slots=True, kw_only=True)
class PredictorResult:
    """Output of the outer predictor stage for one step attempt.

    Produced by ``outer_predictor.py`` and consumed by the outer nonlinear loop
    and step-level diagnostics.
    """

    a_pred: float
    dot_a_pred: float
    first_step_special_case: bool = False
    message: str = ""
    diagnostics: dict[str, object] = field(default_factory=dict)


@dataclass(slots=True, kw_only=True)
class CorrectorResult:
    """Output of the outer corrector stage for one outer iteration.

    Produced by ``outer_corrector.py`` and consumed by
    ``outer_convergence.py`` and step-level diagnostics for the next inner
    entry / transfer chain.
    """

    a_new: float
    dot_a_new: float
    eps_dot_a: float
    relaxed: bool = False
    relaxation_factor: float | None = None
    diagnostics: dict[str, object] = field(default_factory=dict)


@dataclass(slots=True, kw_only=True)
class OuterConvergenceResult:
    """Convergence assessment for one outer iteration.

    Produced by ``outer_convergence.py`` and consumed by the outer-loop driver
    and step-acceptance policy. Convergence is defined on ``eps_dot_a``.
    """

    converged: bool = False
    eps_dot_a: float | None = None
    tolerance: float | None = None
    iteration_index: int = 0
    nonmonotonic_flag: bool = False
    diagnostics: dict[str, object] = field(default_factory=dict)


@dataclass(slots=True, kw_only=True)
class OuterIterationResult:
    """Full record of one outer iteration.

    Produced by the outer iteration driver and consumed by
    ``timestepper.py`` for per-step history and diagnostics.
    """

    outer_iter_index: int
    a_iter: float
    dot_a_iter: float
    inner: InnerSolveResult
    entry_source: InnerEntrySource = InnerEntrySource.ACCEPTED_TIME_LEVEL
    used_transfer: bool = False
    transfer_identity: bool | None = None
    corrector: CorrectorResult | None = None
    convergence: OuterConvergenceResult | None = None
    diagnostics: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.entry_source, InnerEntrySource):
            self.entry_source = InnerEntrySource(self.entry_source)
        if self.entry_source is InnerEntrySource.ACCEPTED_TIME_LEVEL and self.used_transfer:
            raise ValueError("accepted_time_level entry_source requires used_transfer == False")
        if self.entry_source is InnerEntrySource.TRANSFER_FROM_PREVIOUS_OUTER and not self.used_transfer:
            raise ValueError("transfer_from_previous_outer entry_source requires used_transfer == True")


@dataclass(slots=True, kw_only=True)
class StepAcceptanceDecision:
    """Decision returned by the step acceptance / retry policy.

    Produced by ``step_acceptance.py`` and consumed by ``timestepper.py``.
    """

    action: StepAction = StepAction.REJECT
    accepted: bool = False
    rollback_required: bool = True
    dt_current: float | None = None
    dt_next: float | None = None
    reject_reason: str = ""
    failure: FailureInfo = field(default_factory=FailureInfo)
    diagnostics: dict[str, object] = field(default_factory=dict)

    def assert_consistent(self) -> None:
        """Raise ``ValueError`` when acceptance decision fields disagree."""

        if self.action is StepAction.ACCEPT and not self.accepted:
            raise ValueError("ACCEPT action requires accepted == True")
        if self.accepted and self.action is not StepAction.ACCEPT:
            raise ValueError("accepted == True requires action == ACCEPT")
        if self.action is StepAction.ACCEPT and self.rollback_required:
            raise ValueError("ACCEPT action must not require rollback")


@dataclass(slots=True, kw_only=True, init=False)
class StepAdvanceResult:
    """Top-level result for one step attempt.

    Produced by ``timestepper.py`` and consumed by caller-side control flow,
    logging, and restart / forensic logic.

    ``accepted_solution_vec`` is the formal backend-native accepted state
    handle. ``accepted_state_vec`` is retained as a compatibility alias during
    the transition.
    """

    step_id: int
    t_old: float
    t_new_target: float
    dt_attempt: float
    accepted: bool = False
    acceptance: StepAcceptanceDecision = field(default_factory=StepAcceptanceDecision)
    predictor: PredictorResult | None = None
    outer_iterations: list[OuterIterationResult] = field(default_factory=list)
    accepted_solution_vec: object | None = None
    accepted_geometry: object | None = None
    failure: FailureInfo = field(default_factory=FailureInfo)
    diagnostics: dict[str, object] = field(default_factory=dict)

    def __init__(
        self,
        *,
        step_id: int,
        t_old: float,
        t_new_target: float,
        dt_attempt: float,
        accepted: bool = False,
        acceptance: StepAcceptanceDecision | None = None,
        predictor: PredictorResult | None = None,
        outer_iterations: list[OuterIterationResult] | None = None,
        accepted_solution_vec: object | None = None,
        accepted_state_vec: object | None = None,
        accepted_geometry: object | None = None,
        failure: FailureInfo | None = None,
        diagnostics: dict[str, object] | None = None,
    ) -> None:
        if accepted_solution_vec is not None and accepted_state_vec is not None:
            raise ValueError("Provide at most one of accepted_solution_vec or accepted_state_vec")
        self.step_id = int(step_id)
        self.t_old = t_old
        self.t_new_target = t_new_target
        self.dt_attempt = dt_attempt
        self.accepted = bool(accepted)
        self.acceptance = acceptance if acceptance is not None else StepAcceptanceDecision()
        self.predictor = predictor
        self.outer_iterations = list(outer_iterations) if outer_iterations is not None else []
        self.accepted_solution_vec = (
            accepted_solution_vec if accepted_solution_vec is not None else accepted_state_vec
        )
        self.accepted_geometry = accepted_geometry
        self.failure = failure if failure is not None else FailureInfo()
        self.diagnostics = dict(diagnostics) if diagnostics is not None else {}

    @property
    def outer_iter_count(self) -> int:
        """Return the number of completed outer iterations in this attempt."""

        return len(self.outer_iterations)

    @property
    def inner_converged_any(self) -> bool:
        """Return ``True`` if any outer iteration achieved inner convergence."""

        return any(iteration.inner.converged for iteration in self.outer_iterations)

    @property
    def outer_converged(self) -> bool:
        """Return the convergence flag of the last outer iteration, if any."""

        if not self.outer_iterations:
            return False
        last = self.outer_iterations[-1]
        return bool(last.convergence and last.convergence.converged)

    @property
    def accepted_state_vec(self) -> object | None:
        """Compatibility alias for the formal ``accepted_solution_vec`` field."""

        return self.accepted_solution_vec

    def assert_consistent(self) -> None:
        """Raise ``ValueError`` when core step-result invariants are violated."""

        self.acceptance.assert_consistent()
        if self.acceptance.accepted and not self.accepted:
            raise ValueError("acceptance.accepted == True requires step accepted == True")
        if self.acceptance.action is StepAction.ACCEPT and not self.accepted:
            raise ValueError("ACCEPT action requires step accepted == True")
        if self.accepted and not self.acceptance.accepted:
            raise ValueError("accepted step requires acceptance.accepted == True")
        if self.accepted and self.accepted_solution_vec is None:
            raise ValueError("accepted step requires accepted_solution_vec to be present")
        if self.accepted and self.accepted_geometry is None:
            raise ValueError("accepted step requires accepted_geometry to be present")
        if self.accepted and self.failure.failure_class is not FailureClass.NONE:
            raise ValueError("accepted step must not carry a non-NONE failure_class")
        if self.acceptance.action is StepAction.ACCEPT and self.acceptance.rollback_required:
            raise ValueError("ACCEPT action must not require rollback")
        if self.acceptance.action is not StepAction.ACCEPT and self.accepted_solution_vec is not None:
            raise ValueError("non-ACCEPT step must not expose accepted_solution_vec")


__all__ = [
    "CorrectorResult",
    "FailureClass",
    "FailureInfo",
    "GuardFailureReason",
    "InnerEntrySource",
    "InnerSolveResult",
    "InnerSolveStats",
    "OuterConvergenceResult",
    "OuterIterationResult",
    "PredictorResult",
    "ResidualNorms",
    "SolverBackend",
    "StepAcceptanceDecision",
    "StepAction",
    "StepAdvanceResult",
]
