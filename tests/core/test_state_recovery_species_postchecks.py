from __future__ import annotations

import numpy as np
import pytest

from core.state_recovery import StateRecoveryError, _recover_full_mass_fractions
from core.types import RecoveryTemperatureSeeds


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def call_rfmf(species_mass: np.ndarray, mass: np.ndarray, *, eps: float) -> np.ndarray:
    n_full = species_mass.shape[1]
    Y, _ = _recover_full_mass_fractions(
        species_mass,
        mass,
        n_full=n_full,
        species_recovery_eps_abs=eps,
        m_min=1.0e-20,
        Y_sum_tol=1.0e-10,
        Y_hard_tol=1.0e-6,
    )
    return Y


# ---------------------------------------------------------------------------
# Case 1: all-positive input passes unchanged
# ---------------------------------------------------------------------------

def test_all_positive_passes() -> None:
    mass = np.array([1.0, 2.0])
    species_mass = np.array([[0.6, 0.4], [1.2, 0.8]])
    Y = call_rfmf(species_mass, mass, eps=1.0e-14)
    np.testing.assert_allclose(Y[0], [0.6, 0.4], atol=1e-12)
    np.testing.assert_allclose(Y[1], [0.6, 0.4], atol=1e-12)


# ---------------------------------------------------------------------------
# Case 2: tiny negative within eps_abs tolerance → minor fix (clip + renorm)
# ---------------------------------------------------------------------------

def test_minor_fix_tiny_negative_clips_and_renormalizes() -> None:
    eps = 1.0e-10
    mass = np.array([1.0])
    neg_val = -5.0e-11  # magnitude = 5e-11 < mass * eps = 1.0 * 1e-10
    species_mass = np.array([[1.0 - neg_val, neg_val]])  # sum = 1.0
    # After clip: [1.0 - neg_val, 0].  After renorm: [1.0, 0.0]
    Y = call_rfmf(species_mass, mass, eps=eps)
    assert Y[0, 1] == pytest.approx(0.0, abs=1e-15)
    assert Y[0, 0] == pytest.approx(1.0, abs=1e-12)


def test_minor_fix_preserves_mass_conservation() -> None:
    eps = 1.0e-10
    m = 2.0
    mass = np.array([m])
    neg_val = -1.0e-11  # within minor-fix tolerance
    species_mass = np.array([[m - neg_val, neg_val]])
    Y = call_rfmf(species_mass, mass, eps=eps)
    assert abs(np.sum(Y[0]) - 1.0) < 1e-12


# ---------------------------------------------------------------------------
# Case 3: large negative exceeds eps_abs → hard fail
# ---------------------------------------------------------------------------

def test_hard_fail_large_negative_raises() -> None:
    eps = 1.0e-14
    mass = np.array([1.0])
    neg_val = -1.0e-10  # magnitude >> mass * eps = 1e-14
    species_mass = np.array([[1.0 - neg_val, neg_val]])
    with pytest.raises(StateRecoveryError, match="hard-fail"):
        call_rfmf(species_mass, mass, eps=eps)


# ---------------------------------------------------------------------------
# Case 4: negative exactly at tolerance boundary → minor fix succeeds
# ---------------------------------------------------------------------------

def test_minor_fix_at_exact_boundary() -> None:
    eps = 1.0e-10
    mass = np.array([1.0])
    neg_val = -1.0e-10  # exactly at mass * eps
    species_mass = np.array([[1.0 - neg_val, neg_val]])
    Y = call_rfmf(species_mass, mass, eps=eps)
    assert Y[0, 1] == pytest.approx(0.0, abs=1e-15)


# ---------------------------------------------------------------------------
# Case 5: negative just beyond boundary → hard fail
# ---------------------------------------------------------------------------

def test_hard_fail_just_beyond_boundary_raises() -> None:
    eps = 1.0e-10
    mass = np.array([1.0])
    neg_val = -(1.0e-10 + 1.0e-15)  # just beyond mass * eps
    species_mass = np.array([[1.0 - neg_val, neg_val]])
    with pytest.raises(StateRecoveryError, match="hard-fail"):
        call_rfmf(species_mass, mass, eps=eps)


# ---------------------------------------------------------------------------
# Case 6: multi-cell — only the affected row is fixed
# ---------------------------------------------------------------------------

def test_minor_fix_only_affected_row() -> None:
    eps = 1.0e-10
    mass = np.array([1.0, 1.0])
    # Row 0: good; Row 1: tiny negative
    species_mass = np.array([[0.7, 0.3], [1.0 + 1e-11, -1.0e-11]])
    Y = call_rfmf(species_mass, mass, eps=eps)
    np.testing.assert_allclose(Y[0], [0.7, 0.3], atol=1e-12)
    assert Y[1, 1] == pytest.approx(0.0, abs=1e-15)


# ---------------------------------------------------------------------------
# Case 7: single-component liquid always returns ones (unchanged by minor-fix)
# ---------------------------------------------------------------------------

def test_single_component_returns_ones_with_minor_fix() -> None:
    eps = 1.0e-12
    mass = np.array([1.0])
    # Even with exact match, single-component returns ones.
    species_mass = np.array([[1.0]])
    Y, _ = _recover_full_mass_fractions(
        species_mass, mass,
        n_full=1,
        single_component_name="ethanol",
        species_recovery_eps_abs=eps,
        m_min=1.0e-20,
        Y_sum_tol=1.0e-10,
        Y_hard_tol=1.0e-6,
    )
    np.testing.assert_allclose(Y, [[1.0]])


# ---------------------------------------------------------------------------
# Case T4: species sum tolerance scales with cell mass (not max(mass, 1))
# ---------------------------------------------------------------------------

def _rfmf_with_tol(
    species_mass: np.ndarray,
    mass: np.ndarray,
    Y_sum_tol: float,
) -> np.ndarray:
    n_full = species_mass.shape[1]
    Y, _ = _recover_full_mass_fractions(
        species_mass,
        mass,
        n_full=n_full,
        species_recovery_eps_abs=1.0e-30,
        m_min=1.0e-30,
        Y_sum_tol=Y_sum_tol,
        Y_hard_tol=1.0e-3,
    )
    return Y


def test_species_sum_tolerance_scales_with_cell_mass_passes() -> None:
    """diff/mass < Y_sum_tol should pass even for very small mass cells."""
    mass = np.array([1.0e-9])
    tol = 1.0e-8
    # Use 50 % of tolerance to stay safely below the boundary.
    # rel_err = 0.5 * tol < tol → should pass.
    diff = 0.5 * tol * mass[0]  # absolute diff ≈ 5e-18, well within new check
    species_mass = np.array([[mass[0] * 0.7, mass[0] * 0.3 + diff]])
    Y = _rfmf_with_tol(species_mass, mass, Y_sum_tol=tol)
    assert Y is not None
    assert Y.shape == (1, 2)


def test_species_sum_tolerance_scales_with_cell_mass_fails() -> None:
    """diff/mass > Y_sum_tol raises; old max(mass,1) would have incorrectly allowed it.

    With mass=1e-9 and diff = 2*tol*mass = 2e-17:
      - Old check: diff <= tol * max(|mass|, 1) = tol * 1 = 1e-8  → 2e-17 << 1e-8 → PASSES (wrong)
      - New check: rel_err = diff/mass = 2e-8 > tol = 1e-8         → correctly FAILS
    """
    mass = np.array([1.0e-9])
    tol = 1.0e-8
    diff = 2.0 * tol * mass[0]  # rel_err = 2*tol > tol → new check raises
    species_mass = np.array([[mass[0] * 0.7, mass[0] * 0.3 + diff]])
    with pytest.raises(StateRecoveryError, match="per-cell"):
        _rfmf_with_tol(species_mass, mass, Y_sum_tol=tol)


# ---------------------------------------------------------------------------
# Case T5: RecoveryTemperatureSeeds rejects 2-D arrays and ±inf values
# ---------------------------------------------------------------------------

def test_recovery_temperature_seeds_rejects_2d() -> None:
    with pytest.raises(ValueError, match="1-D"):
        RecoveryTemperatureSeeds(T_l=np.ones((2, 3)))


def test_recovery_temperature_seeds_rejects_inf() -> None:
    with pytest.raises(ValueError, match="inf"):
        RecoveryTemperatureSeeds(T_g=np.array([300.0, float("inf")]))


def test_recovery_temperature_seeds_allows_nan() -> None:
    seeds = RecoveryTemperatureSeeds(T_l=np.array([300.0, float("nan"), 350.0]))
    assert seeds.T_l is not None
    assert seeds.T_l.shape == (3,)


def test_recovery_temperature_seeds_array_is_readonly() -> None:
    arr = np.array([300.0, 350.0])
    seeds = RecoveryTemperatureSeeds(T_l=arr)
    assert seeds.T_l is not None
    assert not seeds.T_l.flags.writeable
