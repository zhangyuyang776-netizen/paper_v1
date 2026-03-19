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


@dataclass(slots=True, kw_only=True)
class InnerSolveResult:
    """Result of one fixed-geometry inner solve attempt.

    Produced by ``petsc_snes.py`` and consumed by outer corrector,
    outer-convergence logic, and step acceptance.

    ``state_vec`` is a backend-native global state handle; in the formal PETSc
    path it is expected to be a PETSc Vec rather than a numpy array.
    """

    converged: bool = False
    state_vec: object | None = None
    dot_a_phys: float | None = None
    old_state_on_current_geometry: object | None = None
    stats: InnerSolveStats = field(default_factory=InnerSolveStats)
    failure: FailureInfo = field(default_factory=FailureInfo)
    diagnostics: dict[str, object] = field(default_factory=dict)

    @property
    def has_state(self) -> bool:
        """Return ``True`` when a backend-native state handle is present."""

        return self.state_vec is not None

    def assert_consistent(self) -> None:
        """Raise ``ValueError`` when core inner-result invariants are violated."""

        if self.converged != self.stats.converged:
            raise ValueError("InnerSolveResult.converged must match stats.converged")
        if self.converged and self.failure.failure_class is not FailureClass.NONE:
            raise ValueError("converged inner solve must not carry a non-NONE failure_class")


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
    ``outer_convergence.py`` and step-level diagnostics.
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
    and step-acceptance policy.
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
    corrector: CorrectorResult | None = None
    convergence: OuterConvergenceResult | None = None
    diagnostics: dict[str, object] = field(default_factory=dict)


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


@dataclass(slots=True, kw_only=True)
class StepAdvanceResult:
    """Top-level result for one step attempt.

    Produced by ``timestepper.py`` and consumed by caller-side control flow,
    logging, and restart / forensic logic.

    ``accepted_state_vec`` is a backend-native accepted state handle, expected
    to be a PETSc Vec on the formal PETSc path.
    """

    step_id: int
    t_old: float
    t_new_target: float
    dt_attempt: float
    accepted: bool = False
    acceptance: StepAcceptanceDecision = field(default_factory=StepAcceptanceDecision)
    predictor: PredictorResult | None = None
    outer_iterations: list[OuterIterationResult] = field(default_factory=list)
    accepted_state_vec: object | None = None
    accepted_geometry: object | None = None
    failure: FailureInfo = field(default_factory=FailureInfo)
    diagnostics: dict[str, object] = field(default_factory=dict)

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

    def assert_consistent(self) -> None:
        """Raise ``ValueError`` when core step-result invariants are violated."""

        self.acceptance.assert_consistent()
        if self.acceptance.accepted and not self.accepted:
            raise ValueError("acceptance.accepted == True requires step accepted == True")
        if self.acceptance.action is StepAction.ACCEPT and not self.accepted:
            raise ValueError("ACCEPT action requires step accepted == True")
        if self.accepted and not self.acceptance.accepted:
            raise ValueError("accepted step requires acceptance.accepted == True")
        if self.accepted and self.accepted_state_vec is None:
            raise ValueError("accepted step requires accepted_state_vec to be present")
        if self.accepted and self.accepted_geometry is None:
            raise ValueError("accepted step requires accepted_geometry to be present")
        if self.accepted and self.failure.failure_class is not FailureClass.NONE:
            raise ValueError("accepted step must not carry a non-NONE failure_class")
        if self.acceptance.action is StepAction.ACCEPT and self.acceptance.rollback_required:
            raise ValueError("ACCEPT action must not require rollback")
        if self.acceptance.action is not StepAction.ACCEPT and self.accepted_state_vec is not None:
            raise ValueError("non-ACCEPT step must not expose accepted_state_vec")


__all__ = [
    "CorrectorResult",
    "FailureClass",
    "FailureInfo",
    "GuardFailureReason",
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
