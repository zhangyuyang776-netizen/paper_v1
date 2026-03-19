from __future__ import annotations

import numpy as np
import pytest

from core.types import (
    ConservativeContents,
    GeometryState,
    InterfaceState,
    Mesh1D,
    OldStateOnCurrentGeometry,
    RegionSlices,
    State,
)


def make_mesh_and_state() -> tuple[Mesh1D, State]:
    r_faces = np.array([0.0, 0.1, 0.2, 0.3, 0.5], dtype=np.float64)
    mesh = Mesh1D(
        r_faces=r_faces,
        r_centers=np.array([0.05, 0.15, 0.25, 0.40], dtype=np.float64),
        volumes=(4.0 * np.pi / 3.0) * (r_faces[1:] ** 3 - r_faces[:-1] ** 3),
        face_areas=4.0 * np.pi * r_faces**2,
        dr=np.diff(r_faces),
        region_slices=RegionSlices(
            liq=slice(0, 2),
            gas_near=slice(2, 3),
            gas_far=slice(3, 4),
            gas_all=slice(2, 4),
        ),
        interface_face_index=2,
        interface_cell_liq=1,
        interface_cell_gas=2,
    )
    state = State(
        Tl=np.array([300.0, 305.0], dtype=np.float64),
        Yl_full=np.array([[1.0], [1.0]], dtype=np.float64),
        Tg=np.array([320.0, 330.0], dtype=np.float64),
        Yg_full=np.array([[0.1, 0.21, 0.69], [0.12, 0.20, 0.68]], dtype=np.float64),
        interface=InterfaceState(
            Ts=310.0,
            mpp=0.01,
            Ys_g_full=np.array([0.15, 0.20, 0.65], dtype=np.float64),
            Ys_l_full=np.array([1.0], dtype=np.float64),
        ),
    )
    return mesh, state


def test_conservative_contents_valid() -> None:
    contents = ConservativeContents(
        mass_l=np.array([1.0, 1.1], dtype=np.float64),
        species_mass_l=np.array([[1.0], [1.1]], dtype=np.float64),
        enthalpy_l=np.array([10.0, 11.0], dtype=np.float64),
        mass_g=np.array([0.8, 0.9], dtype=np.float64),
        species_mass_g=np.array([[0.1, 0.2, 0.5], [0.1, 0.2, 0.6]], dtype=np.float64),
        enthalpy_g=np.array([20.0, 21.0], dtype=np.float64),
    )
    assert contents.species_mass_l.shape[0] == contents.mass_l.shape[0]
    assert contents.species_mass_g.shape[0] == contents.mass_g.shape[0]


def test_old_state_on_current_geometry_valid() -> None:
    mesh, state = make_mesh_and_state()
    contents = ConservativeContents(
        mass_l=np.array([1.0, 1.1], dtype=np.float64),
        species_mass_l=np.array([[1.0], [1.1]], dtype=np.float64),
        enthalpy_l=np.array([10.0, 11.0], dtype=np.float64),
        mass_g=np.array([0.8, 0.9], dtype=np.float64),
        species_mass_g=np.array([[0.1, 0.2, 0.5], [0.1, 0.2, 0.6]], dtype=np.float64),
        enthalpy_g=np.array([20.0, 21.0], dtype=np.float64),
    )
    old_state = OldStateOnCurrentGeometry(
        contents=contents,
        state=state,
        geometry=GeometryState(
            t=0.0,
            dt=1.0e-7,
            a=5.0e-5,
            dot_a=-1.0e-3,
            r_end=5.0e-3,
            step_index=0,
            outer_iter_index=0,
        ),
        mesh=mesh,
    )
    assert old_state.contents.n_liq_cells == old_state.state.n_liq_cells


def test_conservative_contents_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        ConservativeContents(
            mass_l=np.array([1.0, 1.1], dtype=np.float64),
            species_mass_l=np.array([[1.0]], dtype=np.float64),
            enthalpy_l=np.array([10.0, 11.0], dtype=np.float64),
            mass_g=np.array([0.8, 0.9], dtype=np.float64),
            species_mass_g=np.array([[0.1, 0.2], [0.1, 0.2]], dtype=np.float64),
            enthalpy_g=np.array([20.0, 21.0], dtype=np.float64),
        )
