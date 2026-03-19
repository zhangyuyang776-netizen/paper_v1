from __future__ import annotations

import numpy as np
import pytest

from core.types import GeometryState, Mesh1D, RegionSlices


def make_region_slices() -> RegionSlices:
    return RegionSlices(
        liq=slice(0, 2),
        gas_near=slice(2, 3),
        gas_far=slice(3, 4),
        gas_all=slice(2, 4),
    )


def make_mesh() -> Mesh1D:
    r_faces = np.array([0.0, 0.1, 0.2, 0.3, 0.5], dtype=np.float64)
    r_centers = np.array([0.05, 0.15, 0.25, 0.40], dtype=np.float64)
    volumes = (4.0 * np.pi / 3.0) * (r_faces[1:] ** 3 - r_faces[:-1] ** 3)
    face_areas = 4.0 * np.pi * r_faces**2
    dr = np.diff(r_faces)
    return Mesh1D(
        r_faces=r_faces,
        r_centers=r_centers,
        volumes=volumes,
        face_areas=face_areas,
        dr=dr,
        region_slices=make_region_slices(),
        interface_face_index=2,
        interface_cell_liq=1,
        interface_cell_gas=2,
    )


def test_geometry_state_valid() -> None:
    geom = GeometryState(
        t=0.0,
        dt=1.0e-7,
        a=5.0e-5,
        dot_a=-1.0e-3,
        r_end=5.0e-3,
        step_index=0,
        outer_iter_index=0,
    )
    assert geom.a > 0.0
    assert geom.dt > 0.0
    assert geom.is_first_step() is True


def test_mesh1d_valid() -> None:
    mesh = make_mesh()
    assert mesh.n_faces == len(mesh.r_faces)
    assert mesh.n_cells == len(mesh.r_centers)
    assert np.all(mesh.volumes > 0.0)
    assert np.all(np.diff(mesh.r_faces) > 0.0)


def test_mesh1d_nonmonotonic_faces_raises() -> None:
    with pytest.raises(ValueError):
        Mesh1D(
            r_faces=np.array([0.0, 0.1, 0.09, 0.2], dtype=np.float64),
            r_centers=np.array([0.05, 0.095, 0.145], dtype=np.float64),
            volumes=np.array([1.0, 1.0, 1.0], dtype=np.float64),
            face_areas=np.array([0.0, 1.0, 1.0, 1.0], dtype=np.float64),
            dr=np.array([0.1, -0.01, 0.11], dtype=np.float64),
            region_slices=RegionSlices(
                liq=slice(0, 1),
                gas_near=slice(1, 2),
                gas_far=slice(2, 3),
                gas_all=slice(1, 3),
            ),
        )


def test_mesh1d_inconsistent_lengths_raises() -> None:
    with pytest.raises(ValueError):
        Mesh1D(
            r_faces=np.array([0.0, 0.1, 0.2], dtype=np.float64),
            r_centers=np.array([0.05, 0.15], dtype=np.float64),
            volumes=np.array([1.0], dtype=np.float64),
            face_areas=np.array([0.0, 1.0, 1.0], dtype=np.float64),
            dr=np.array([0.1, 0.1], dtype=np.float64),
            region_slices=RegionSlices(
                liq=slice(0, 1),
                gas_near=slice(1, 2),
                gas_far=slice(2, 2),
                gas_all=slice(1, 2),
            ),
        )
