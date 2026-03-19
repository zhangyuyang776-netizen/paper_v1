from __future__ import annotations

import numpy as np
import pytest

from core.types import (
    ConservativeContents,
    GeometryState,
    InterfaceState,
    Mesh1D,
    OldStateOnCurrentGeometry,
    OuterIterState,
    Props,
    RegionSlices,
    SpeciesMaps,
    State,
)


def _build_base_mesh_kwargs() -> dict[str, object]:
    return {
        "r_faces": np.array([0.0, 1.0, 2.0, 3.0]),
        "r_centers": np.array([0.5, 1.5, 2.5]),
        "volumes": np.array([1.0, 1.0, 1.0]),
        "face_areas": np.array([0.0, 1.0, 1.0, 1.0]),
        "dr": np.array([1.0, 1.0, 1.0]),
        "interface_face_index": 1,
        "interface_cell_liq": 0,
        "interface_cell_gas": 1,
    }


def _build_region() -> RegionSlices:
    return RegionSlices(
        liq=slice(0, 1),
        gas_near=slice(1, 3),
        gas_far=slice(3, 3),
        gas_all=slice(1, 3),
    )


def _build_mesh_a() -> Mesh1D:
    return Mesh1D(
        **_build_base_mesh_kwargs(),
        region_slices=_build_region(),
    )


def _build_mesh_b_same_counts_but_different_geometry() -> Mesh1D:
    return Mesh1D(
        r_faces=np.array([0.0, 1.2, 2.0, 3.0]),
        r_centers=np.array([0.6, 1.6, 2.5]),
        volumes=np.array([1.2, 0.8, 1.0]),
        face_areas=np.array([0.0, 1.44, 1.0, 1.0]),
        dr=np.array([1.2, 0.8, 1.0]),
        region_slices=_build_region(),
        interface_face_index=1,
        interface_cell_liq=0,
        interface_cell_gas=1,
    )


def _build_state() -> State:
    interface = InterfaceState(
        Ts=400.0,
        mpp=0.0,
        Ys_g_full=np.array([0.2, 0.8]),
        Ys_l_full=np.array([1.0]),
    )
    return State(
        Tl=np.array([350.0]),
        Yl_full=np.array([[1.0]]),
        Tg=np.array([500.0, 510.0]),
        Yg_full=np.array([[0.2, 0.8], [0.21, 0.79]]),
        interface=interface,
    )


def _build_contents() -> ConservativeContents:
    return ConservativeContents(
        mass_l=np.array([1.0]),
        species_mass_l=np.array([[1.0]]),
        enthalpy_l=np.array([100.0]),
        mass_g=np.array([1.0, 1.0]),
        species_mass_g=np.array([[0.2, 0.8], [0.21, 0.79]]),
        enthalpy_g=np.array([200.0, 210.0]),
    )


def test_region_slices_rejects_liq_not_starting_at_zero() -> None:
    with pytest.raises(ValueError, match="liq slice must start at 0"):
        RegionSlices(
            liq=slice(1, 2),
            gas_near=slice(2, 3),
            gas_far=slice(3, 3),
            gas_all=slice(2, 3),
        )


def test_mesh1d_rejects_uncovered_tail_cells() -> None:
    region = RegionSlices(
        liq=slice(0, 1),
        gas_near=slice(1, 2),
        gas_far=slice(2, 2),
        gas_all=slice(1, 2),
    )
    kwargs = _build_base_mesh_kwargs()
    kwargs["region_slices"] = region

    with pytest.raises(ValueError, match="gas_all slice must end at n_cells"):
        Mesh1D(**kwargs)


def test_species_maps_rejects_extra_full_to_reduced_alias() -> None:
    with pytest.raises(
        ValueError,
        match="full_to_reduced marks extra mapped species|do not select the same full species",
    ):
        SpeciesMaps(
            liq_full_names=("A", "B", "C"),
            liq_active_names=("A",),
            liq_closure_name="C",
            gas_full_names=("N2", "O2"),
            gas_active_names=("O2",),
            gas_closure_name="N2",
            liq_full_to_reduced=np.array([0, 0, -1], dtype=np.int64),
            liq_reduced_to_full=np.array([0], dtype=np.int64),
            gas_full_to_reduced=np.array([-1, 0], dtype=np.int64),
            gas_reduced_to_full=np.array([1], dtype=np.int64),
            liq_full_to_gas_full=np.array([1, 1, 1], dtype=np.int64),
        )


def test_outer_iter_state_rejects_different_mesh_with_same_counts() -> None:
    geom = GeometryState(
        t=0.0,
        dt=1.0e-6,
        a=1.0e-4,
        dot_a=0.0,
        r_end=1.0e-2,
        step_index=0,
        outer_iter_index=0,
    )

    old_state = OldStateOnCurrentGeometry(
        contents=_build_contents(),
        state=_build_state(),
        geometry=geom,
        mesh=_build_mesh_a(),
    )

    with pytest.raises(ValueError, match="must match current outer mesh exactly"):
        OuterIterState(
            geometry=geom,
            mesh=_build_mesh_b_same_counts_but_different_geometry(),
            old_state_current_geom=old_state,
            predicted_from_accepted=True,
        )


def test_state_rejects_nan_in_temperature_array() -> None:
    interface = InterfaceState(
        Ts=400.0,
        mpp=0.0,
        Ys_g_full=np.array([0.2, 0.8]),
        Ys_l_full=np.array([1.0]),
    )

    with pytest.raises(ValueError, match="Tl must contain only finite values"):
        State(
            Tl=np.array([300.0, np.nan]),
            Yl_full=np.array([[1.0], [1.0]]),
            Tg=np.array([500.0]),
            Yg_full=np.array([[0.2, 0.8]]),
            interface=interface,
        )


def test_conservative_contents_rejects_inf() -> None:
    with pytest.raises(ValueError, match="mass_g must contain only finite values"):
        ConservativeContents(
            mass_l=np.array([1.0]),
            species_mass_l=np.array([[1.0]]),
            enthalpy_l=np.array([100.0]),
            mass_g=np.array([1.0, np.inf]),
            species_mass_g=np.array([[0.2, 0.8], [0.3, 0.7]]),
            enthalpy_g=np.array([200.0, 210.0]),
        )


def test_species_maps_mapping_arrays_are_read_only() -> None:
    sm = SpeciesMaps(
        liq_full_names=("A",),
        liq_active_names=(),
        liq_closure_name=None,
        gas_full_names=("N2", "O2"),
        gas_active_names=("O2",),
        gas_closure_name="N2",
        liq_full_to_reduced=np.array([-1], dtype=np.int64),
        liq_reduced_to_full=np.array([], dtype=np.int64),
        gas_full_to_reduced=np.array([-1, 0], dtype=np.int64),
        gas_reduced_to_full=np.array([1], dtype=np.int64),
        liq_full_to_gas_full=np.array([1], dtype=np.int64),
    )

    with pytest.raises(ValueError):
        sm.gas_full_to_reduced[0] = 123


def test_props_rejects_dg_xg_full_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="Dg and Xg_full must have the same shape"):
        Props(
            rho_g=np.array([1.0, 1.1]),
            Dg=np.array([[1.0e-5, 2.0e-5], [1.1e-5, 2.1e-5]]),
            Xg_full=np.array([[0.2, 0.8, 0.0], [0.3, 0.7, 0.0]]),
        )
