from __future__ import annotations

import numpy as np

from physics.energy_flux import build_energy_flux_package, compute_total_energy_flux_density


def test_conductive_only_flux_matches_minus_k_gradT() -> None:
    k = np.array([0.1, 0.2], dtype=np.float64)
    grad_T = np.array([5.0, -3.0], dtype=np.float64)
    J = np.zeros((2, 3), dtype=np.float64)
    h = np.array([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]], dtype=np.float64)

    q_cond, q_species_diff, q_total = compute_total_energy_flux_density(k, grad_T, J, h)

    assert np.allclose(q_cond, -k * grad_T)
    assert np.allclose(q_species_diff, 0.0)
    assert np.allclose(q_total, q_cond)


def test_species_diffusion_enthalpy_flux_is_sum_of_J_times_h() -> None:
    J = np.array([[1.0, -2.0], [3.0, 4.0]], dtype=np.float64)
    h = np.array([[10.0, 20.0], [5.0, 6.0]], dtype=np.float64)

    _, q_species_diff, _ = compute_total_energy_flux_density(
        np.array([1.0, 1.0], dtype=np.float64),
        np.array([0.0, 0.0], dtype=np.float64),
        J,
        h,
    )

    assert np.allclose(q_species_diff, np.array([1.0 * 10.0 + (-2.0) * 20.0, 3.0 * 5.0 + 4.0 * 6.0], dtype=np.float64))


def test_total_heat_flux_is_qcond_plus_qdiff() -> None:
    k = np.array([0.2, 0.3], dtype=np.float64)
    grad_T = np.array([2.0, -4.0], dtype=np.float64)
    J = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    h = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float64)

    q_cond, q_species_diff, q_total = compute_total_energy_flux_density(k, grad_T, J, h)

    assert np.allclose(q_total, q_cond + q_species_diff)


def test_nonzero_J_and_zero_gradT_still_gives_nonzero_total_heat_flux() -> None:
    J = np.array([[1.5, -0.5]], dtype=np.float64)
    h = np.array([[100.0, 200.0]], dtype=np.float64)

    q_cond, q_species_diff, q_total = compute_total_energy_flux_density(
        np.array([0.5], dtype=np.float64),
        np.array([0.0], dtype=np.float64),
        J,
        h,
    )

    assert np.allclose(q_cond, 0.0)
    assert not np.allclose(q_species_diff, 0.0)
    assert np.allclose(q_total, q_species_diff)


def test_zero_J_reduces_to_pure_conduction() -> None:
    k = np.array([0.5], dtype=np.float64)
    grad_T = np.array([4.0], dtype=np.float64)
    J = np.zeros((1, 2), dtype=np.float64)
    h = np.array([[50.0, 60.0]], dtype=np.float64)

    q_cond, q_species_diff, q_total = compute_total_energy_flux_density(k, grad_T, J, h)

    assert np.allclose(q_species_diff, 0.0)
    assert np.allclose(q_total, q_cond)


def test_shape_mismatch_between_J_and_h_raises() -> None:
    try:
        compute_total_energy_flux_density(
            np.array([1.0], dtype=np.float64),
            np.array([2.0], dtype=np.float64),
            np.array([[1.0, 2.0]], dtype=np.float64),
            np.array([[3.0, 4.0, 5.0]], dtype=np.float64),
        )
    except Exception as exc:
        assert "column count" in str(exc)
    else:
        raise AssertionError("expected shape mismatch failure")


def test_face_count_mismatch_raises() -> None:
    try:
        build_energy_flux_package(
            face_indices=np.array([1, 2], dtype=np.int64),
            k_face=np.array([1.0], dtype=np.float64),
            grad_T_face=np.array([2.0, 3.0], dtype=np.float64),
            J_diff_full=np.array([[1.0], [2.0]], dtype=np.float64),
            h_face_full=np.array([[3.0], [4.0]], dtype=np.float64),
        )
    except Exception as exc:
        assert "length 2" in str(exc)
    else:
        raise AssertionError("expected face count mismatch failure")


def test_negative_or_nonfinite_k_raises() -> None:
    for bad_k in (
        np.array([-1.0], dtype=np.float64),
        np.array([np.nan], dtype=np.float64),
    ):
        try:
            compute_total_energy_flux_density(
                bad_k,
                np.array([1.0], dtype=np.float64),
                np.array([[0.0]], dtype=np.float64),
                np.array([[1.0]], dtype=np.float64),
            )
        except Exception:
            pass
        else:
            raise AssertionError("expected invalid conductivity to be rejected")


def test_area_optional_and_when_given_returns_Q() -> None:
    pkg_no_area = build_energy_flux_package(
        face_indices=np.array([1], dtype=np.int64),
        k_face=np.array([2.0], dtype=np.float64),
        grad_T_face=np.array([3.0], dtype=np.float64),
        J_diff_full=np.array([[4.0, 5.0]], dtype=np.float64),
        h_face_full=np.array([[6.0, 7.0]], dtype=np.float64),
    )
    assert pkg_no_area.area_face is None
    assert pkg_no_area.Q_cond is None
    assert pkg_no_area.Q_species_diff is None
    assert pkg_no_area.Q_total is None

    pkg = build_energy_flux_package(
        face_indices=np.array([1], dtype=np.int64),
        k_face=np.array([2.0], dtype=np.float64),
        grad_T_face=np.array([3.0], dtype=np.float64),
        J_diff_full=np.array([[4.0, 5.0]], dtype=np.float64),
        h_face_full=np.array([[6.0, 7.0]], dtype=np.float64),
        area_face=np.array([10.0], dtype=np.float64),
    )
    assert np.allclose(pkg.Q_cond, pkg.q_cond * 10.0)
    assert np.allclose(pkg.Q_species_diff, pkg.q_species_diff * 10.0)
    assert np.allclose(pkg.Q_total, pkg.q_total * 10.0)


def test_single_face_case_supported() -> None:
    pkg = build_energy_flux_package(
        face_indices=np.array([99], dtype=np.int64),
        k_face=np.array([0.8], dtype=np.float64),
        grad_T_face=np.array([1.5], dtype=np.float64),
        J_diff_full=np.array([[0.1, -0.1]], dtype=np.float64),
        h_face_full=np.array([[100.0, 200.0]], dtype=np.float64),
        area_face=np.array([2.0], dtype=np.float64),
    )

    assert pkg.face_indices.shape == (1,)
    assert pkg.q_total.shape == (1,)
    assert pkg.Q_total.shape == (1,)
