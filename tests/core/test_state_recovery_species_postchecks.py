from __future__ import annotations

import numpy as np
import pytest

from core.state_recovery import StateRecoveryError, _recover_full_mass_fractions


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def call_rfmf(species_mass: np.ndarray, mass: np.ndarray, *, eps: float) -> np.ndarray:
    n_full = species_mass.shape[1]
    return _recover_full_mass_fractions(
        species_mass,
        mass,
        n_full=n_full,
        species_recovery_eps_abs=eps,
    )


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
    Y = _recover_full_mass_fractions(
        species_mass, mass,
        n_full=1,
        single_component_name="ethanol",
        species_recovery_eps_abs=eps,
    )
    np.testing.assert_allclose(Y, [[1.0]])
