from __future__ import annotations

import numpy as np

from core.types import Mesh1D, RegionSlices
from physics.flux_convective import (
    build_gas_farfield_boundary_convective_flux_scalar,
    build_gas_farfield_boundary_convective_flux_vector,
    build_gas_internal_convective_flux_scalar,
    build_gas_internal_convective_flux_vector,
    build_liquid_center_boundary_convective_flux,
    build_liquid_internal_convective_flux_scalar,
    build_liquid_internal_convective_flux_vector,
)


def _make_stub_mesh() -> Mesh1D:
    return Mesh1D(
        r_faces=np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float64),
        r_centers=np.array([0.5, 1.5, 2.5, 3.5, 4.5, 5.5], dtype=np.float64),
        volumes=np.ones(6, dtype=np.float64),
        face_areas=np.array([10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0], dtype=np.float64),
        dr=np.ones(6, dtype=np.float64),
        region_slices=RegionSlices(
            liq=slice(0, 3),
            gas_near=slice(3, 4),
            gas_far=slice(4, 6),
            gas_all=slice(3, 6),
        ),
        interface_face_index=3,
        interface_cell_liq=2,
        interface_cell_gas=3,
    )


def test_relative_velocity_is_u_minus_vc_via_flux_package() -> None:
    mesh = _make_stub_mesh()
    phi = np.array([10.0, 20.0, 30.0], dtype=np.float64)
    u_abs = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    vc = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float64)

    pkg = build_liquid_internal_convective_flux_scalar(mesh, phi, u_abs, vc)

    assert np.allclose(pkg.u_rel_face, np.array([1.8, 2.7], dtype=np.float64))


def test_internal_scalar_upwind_uses_left_when_u_rel_positive() -> None:
    mesh = _make_stub_mesh()
    phi = np.array([10.0, 20.0, 30.0], dtype=np.float64)
    u_abs = np.array([0.0, 2.0, 3.0, 0.0], dtype=np.float64)
    vc = np.array([0.0, 0.5, 0.5, 0.0], dtype=np.float64)

    pkg = build_liquid_internal_convective_flux_scalar(mesh, phi, u_abs, vc)

    assert np.array_equal(pkg.upwind_is_left, np.array([True, True]))
    assert np.allclose(pkg.phi_upwind, np.array([10.0, 20.0], dtype=np.float64))
    assert np.allclose(pkg.flux, np.array([11.0 * 1.5 * 10.0, 12.0 * 2.5 * 20.0], dtype=np.float64))


def test_internal_scalar_upwind_uses_right_when_u_rel_negative() -> None:
    mesh = _make_stub_mesh()
    phi = np.array([10.0, 20.0, 30.0], dtype=np.float64)
    u_abs = np.array([0.0, -1.0, -2.0, 0.0], dtype=np.float64)
    vc = np.array([0.0, 0.5, 0.5, 0.0], dtype=np.float64)

    pkg = build_liquid_internal_convective_flux_scalar(mesh, phi, u_abs, vc)

    assert np.array_equal(pkg.upwind_is_left, np.array([False, False]))
    assert np.allclose(pkg.phi_upwind, np.array([20.0, 30.0], dtype=np.float64))
    assert np.allclose(pkg.flux, np.array([11.0 * (-1.5) * 20.0, 12.0 * (-2.5) * 30.0], dtype=np.float64))


def test_internal_vector_upwind_uses_same_side_for_all_components() -> None:
    mesh = _make_stub_mesh()
    phi = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]], dtype=np.float64)
    u_abs = np.array([0.0, 2.0, -2.0, 0.0], dtype=np.float64)
    vc = np.array([0.0, 0.5, 0.5, 0.0], dtype=np.float64)

    pkg = build_liquid_internal_convective_flux_vector(mesh, phi, u_abs, vc)

    assert np.array_equal(pkg.upwind_is_left, np.array([True, False]))
    assert np.allclose(pkg.phi_upwind, np.array([[1.0, 10.0], [3.0, 30.0]], dtype=np.float64))
    assert np.allclose(
        pkg.flux,
        np.array(
            [
                [11.0 * 1.5 * 1.0, 11.0 * 1.5 * 10.0],
                [12.0 * (-2.5) * 3.0, 12.0 * (-2.5) * 30.0],
            ],
            dtype=np.float64,
        ),
    )


def test_zero_relative_velocity_gives_zero_flux() -> None:
    mesh = _make_stub_mesh()
    phi = np.array([10.0, 20.0, 30.0], dtype=np.float64)
    u_abs = np.array([0.0, 1.0, 2.0, 0.0], dtype=np.float64)
    vc = np.array([0.0, 1.0, 2.0, 0.0], dtype=np.float64)

    pkg = build_liquid_internal_convective_flux_scalar(mesh, phi, u_abs, vc)

    assert np.allclose(pkg.u_rel_face, 0.0)
    assert np.allclose(pkg.flux, 0.0)


def test_liquid_center_boundary_convective_flux_is_zero() -> None:
    pkg_scalar = build_liquid_center_boundary_convective_flux(12.0)
    pkg_vector = build_liquid_center_boundary_convective_flux(np.array([1.0, 2.0], dtype=np.float64))

    assert pkg_scalar.face_index == 0
    assert np.allclose(pkg_scalar.flux, 0.0)
    assert np.allclose(pkg_vector.flux, np.array([0.0, 0.0], dtype=np.float64))


def test_gas_farfield_outflow_uses_last_cell_state() -> None:
    mesh = _make_stub_mesh()
    pkg = build_gas_farfield_boundary_convective_flux_scalar(
        mesh,
        phi_cell_last=9.0,
        phi_bc=99.0,
        u_abs_face_far=3.0,
        vc_face_far=1.0,
    )

    assert pkg.upwind_from_cell is True
    assert np.allclose(pkg.phi_face, np.array([9.0], dtype=np.float64))
    assert np.allclose(pkg.flux, np.array([16.0 * 2.0 * 9.0], dtype=np.float64))


def test_gas_farfield_inflow_uses_boundary_state() -> None:
    mesh = _make_stub_mesh()
    pkg = build_gas_farfield_boundary_convective_flux_vector(
        mesh,
        phi_cell_last=np.array([9.0, 19.0], dtype=np.float64),
        phi_bc=np.array([99.0, 199.0], dtype=np.float64),
        u_abs_face_far=1.0,
        vc_face_far=3.0,
    )

    assert pkg.upwind_from_cell is False
    assert np.allclose(pkg.phi_face, np.array([99.0, 199.0], dtype=np.float64))
    assert np.allclose(pkg.flux, 16.0 * (-2.0) * np.array([99.0, 199.0], dtype=np.float64))


def test_liquid_internal_faces_exclude_center_and_interface() -> None:
    mesh = _make_stub_mesh()
    pkg = build_liquid_internal_convective_flux_scalar(
        mesh,
        np.array([1.0, 2.0, 3.0], dtype=np.float64),
        np.array([0.0, 1.0, 1.0, 0.0], dtype=np.float64),
        np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64),
    )

    assert 0 not in pkg.face_indices
    assert mesh.interface_face_index not in pkg.face_indices
    assert np.array_equal(pkg.face_indices, np.array([1, 2], dtype=np.int64))


def test_gas_internal_faces_exclude_interface_and_farfield() -> None:
    mesh = _make_stub_mesh()
    pkg = build_gas_internal_convective_flux_scalar(
        mesh,
        np.array([4.0, 5.0, 6.0], dtype=np.float64),
        np.array([0.0, 1.0, 1.0, 0.0], dtype=np.float64),
        np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64),
    )

    assert mesh.interface_face_index not in pkg.face_indices
    assert (mesh.n_faces - 1) not in pkg.face_indices
    assert np.array_equal(pkg.face_indices, np.array([4, 5], dtype=np.int64))


def test_scalar_and_vector_field_shapes_are_validated() -> None:
    mesh = _make_stub_mesh()

    try:
        build_liquid_internal_convective_flux_scalar(
            mesh,
            np.array([1.0, 2.0], dtype=np.float64),
            np.array([0.0, 1.0, 1.0, 0.0], dtype=np.float64),
            np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64),
        )
    except Exception as exc:
        assert "connectivity exceeds scalar phi_cell length" in str(exc)
    else:
        raise AssertionError("expected scalar shape validation failure")

    try:
        build_gas_internal_convective_flux_vector(
            mesh,
            np.array([1.0, 2.0, 3.0], dtype=np.float64),
            np.array([0.0, 1.0, 1.0, 0.0], dtype=np.float64),
            np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64),
        )
    except Exception as exc:
        assert "must be a 2D float array" in str(exc)
    else:
        raise AssertionError("expected vector shape validation failure")
