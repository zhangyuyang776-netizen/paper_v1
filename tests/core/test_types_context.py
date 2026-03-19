from __future__ import annotations

import numpy as np

from core.types import (
    ConservativeContents,
    GeometryState,
    InterfaceState,
    Mesh1D,
    OldStateOnCurrentGeometry,
    OuterIterState,
    RegionSlices,
    State,
    StepContext,
)


def make_mesh_state_old() -> tuple[Mesh1D, State, OldStateOnCurrentGeometry]:
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
    geometry = GeometryState(
        t=0.0,
        dt=1.0e-7,
        a=5.0e-5,
        dot_a=-1.0e-3,
        r_end=5.0e-3,
        step_index=0,
        outer_iter_index=0,
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
    old_state = OldStateOnCurrentGeometry(
        contents=ConservativeContents(
            mass_l=np.array([1.0, 1.1], dtype=np.float64),
            species_mass_l=np.array([[1.0], [1.1]], dtype=np.float64),
            enthalpy_l=np.array([10.0, 11.0], dtype=np.float64),
            mass_g=np.array([0.8, 0.9], dtype=np.float64),
            species_mass_g=np.array([[0.1, 0.2, 0.5], [0.1, 0.2, 0.6]], dtype=np.float64),
            enthalpy_g=np.array([20.0, 21.0], dtype=np.float64),
        ),
        state=state,
        geometry=geometry,
        mesh=mesh,
    )
    return mesh, state, old_state


def test_outer_iter_state_valid() -> None:
    mesh, _, old_state = make_mesh_state_old()
    outer_iter = OuterIterState(
        geometry=old_state.geometry,
        mesh=mesh,
        old_state_current_geom=old_state,
        predicted_from_accepted=True,
    )
    assert outer_iter.old_state_current_geom is not None


def test_step_context_valid() -> None:
    mesh, state, old_state = make_mesh_state_old()
    ctx = StepContext(
        accepted_state=state,
        accepted_geometry=old_state.geometry,
        accepted_mesh=mesh,
        dt_try=1.0e-7,
        step_index=0,
        retry_count_for_current_state=0,
        last_failure_class=None,
        accepted_state_id="state-0",
    )
    assert ctx.retry_count_for_current_state == 0
    assert ctx.last_failure_class is None
