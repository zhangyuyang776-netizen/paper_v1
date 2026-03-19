from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from assembly.residual_interface import (
    InterfaceResidualAssemblyError,
    assemble_interface_residual,
)
from core.layout import UnknownLayout
from core.types import SpeciesMaps
from physics.interface_energy import build_interface_energy_residual_package
from physics.interface_face import InterfaceFacePackage
from physics.interface_mass import build_interface_mass_residual_package


def _make_species_maps_single() -> SpeciesMaps:
    return SpeciesMaps(
        liq_full_names=("ethanol",),
        liq_active_names=(),
        liq_closure_name="ethanol",
        gas_full_names=("C2H5OH", "N2"),
        gas_active_names=("C2H5OH",),
        gas_closure_name="N2",
        liq_full_to_reduced=np.asarray([-1], dtype=np.int64),
        liq_reduced_to_full=np.asarray([], dtype=np.int64),
        gas_full_to_reduced=np.asarray([0, -1], dtype=np.int64),
        gas_reduced_to_full=np.asarray([0], dtype=np.int64),
        liq_full_to_gas_full=np.asarray([0], dtype=np.int64),
    )


def _make_species_maps_multi() -> SpeciesMaps:
    return SpeciesMaps(
        liq_full_names=("ethanol", "water"),
        liq_active_names=("ethanol",),
        liq_closure_name="water",
        gas_full_names=("C2H5OH", "H2O", "O2", "N2"),
        gas_active_names=("C2H5OH", "H2O", "O2"),
        gas_closure_name="N2",
        liq_full_to_reduced=np.asarray([0, -1], dtype=np.int64),
        liq_reduced_to_full=np.asarray([0], dtype=np.int64),
        gas_full_to_reduced=np.asarray([0, 1, 2, -1], dtype=np.int64),
        gas_reduced_to_full=np.asarray([0, 1, 2], dtype=np.int64),
        liq_full_to_gas_full=np.asarray([0, 1], dtype=np.int64),
    )


def _make_layout_single() -> UnknownLayout:
    return UnknownLayout(
        unknowns_profile="U_A",
        n_liq_cells=2,
        n_gas_cells=1,
        n_liq_red=0,
        n_gas_red=1,
        liq_cell_width=1,
        gas_cell_width=2,
        n_if_unknowns=3,
        liq_block=slice(0, 2),
        if_block=slice(2, 5),
        gas_block=slice(5, 7),
        total_size=7,
        liq_temperature_slice=slice(0, 2, 1),
        liq_species_local_slice=slice(1, 1),
        if_temperature_slice=slice(2, 3),
        if_gas_species_slice=slice(3, 4),
        if_mpp_slice=slice(4, 5),
        if_liq_species_slice=slice(5, 5),
        gas_temperature_slice=slice(5, 7, 2),
        gas_species_local_slice=slice(1, 2),
    )


def _make_layout_multi() -> UnknownLayout:
    return UnknownLayout(
        unknowns_profile="U_B",
        n_liq_cells=2,
        n_gas_cells=1,
        n_liq_red=1,
        n_gas_red=3,
        liq_cell_width=2,
        gas_cell_width=4,
        n_if_unknowns=6,
        liq_block=slice(0, 4),
        if_block=slice(4, 10),
        gas_block=slice(10, 14),
        total_size=14,
        liq_temperature_slice=slice(0, 4, 2),
        liq_species_local_slice=slice(1, 2),
        if_temperature_slice=slice(4, 5),
        if_gas_species_slice=slice(5, 8),
        if_mpp_slice=slice(8, 9),
        if_liq_species_slice=slice(9, 10),
        gas_temperature_slice=slice(10, 14, 4),
        gas_species_local_slice=slice(1, 4),
    )


def _make_run_cfg(species_maps: SpeciesMaps) -> SimpleNamespace:
    return SimpleNamespace(species_maps=species_maps)


def _make_iface_pkg_single() -> InterfaceFacePackage:
    return InterfaceFacePackage(
        r_s=1.0e-3,
        area_s=1.0e-5,
        dr_l_s=5.0e-5,
        dr_g_s=7.5e-5,
        dot_a_frozen=-0.01,
        Tl_last=300.0,
        Tg_first=320.0,
        Yl_last_full=np.asarray([1.0], dtype=np.float64),
        Yg_first_full=np.asarray([0.2, 0.8], dtype=np.float64),
        Xg_first_full=np.asarray([0.25, 0.75], dtype=np.float64),
        Ts=310.0,
        Ys_l_full=np.asarray([1.0], dtype=np.float64),
        Ys_g_full=np.asarray([0.3, 0.7], dtype=np.float64),
        Xs_g_full=np.asarray([0.35, 0.65], dtype=np.float64),
        mpp=0.1,
        rho_s_l=800.0,
        rho_s_g=1.2,
        h_s_l=1.0e5,
        h_s_g=2.0e5,
        h_liq_species_s_full=np.asarray([1.1e5], dtype=np.float64),
        h_gas_species_s_full=np.asarray([2.1e5, 1.4e5], dtype=np.float64),
        k_s_l=0.15,
        k_s_g=0.04,
        D_s_l_full=np.asarray([0.0], dtype=np.float64),
        D_s_g_full=np.asarray([1.0e-5, 1.1e-5], dtype=np.float64),
        dTdr_l_s=1.0,
        dTdr_g_s=2.0,
        dYdr_l_s_full=np.asarray([0.0], dtype=np.float64),
        dXdr_g_s_full=np.asarray([1.0, -1.0], dtype=np.float64),
        J_l_full=np.asarray([0.0], dtype=np.float64),
        J_g_full=np.asarray([0.0, 0.0], dtype=np.float64),
        Vd0_g_full=np.asarray([0.0, 0.0], dtype=np.float64),
        Vcd_g=0.0,
        N_l_full=np.asarray([0.0], dtype=np.float64),
        N_g_full=np.asarray([0.2, -0.2], dtype=np.float64),
        q_l_s=10.0,
        q_g_s=20.0,
        q_l_cond=6.0,
        q_g_cond=8.0,
        q_l_species_diff=4.0,
        q_g_species_diff=12.0,
        E_l_s=5.0,
        E_g_s=9.0,
        G_g_if_abs=-1.12e-6,
        Yeq_g_cond_full=np.asarray([0.25, 0.75], dtype=np.float64),
        gamma_cond_full=np.asarray([1.0], dtype=np.float64),
        eps_mass_gas_kinematic=0.0,
    )


def _make_iface_pkg_multi() -> InterfaceFacePackage:
    return InterfaceFacePackage(
        r_s=1.0e-3,
        area_s=float(4.0 * np.pi * 1.0e-6),
        dr_l_s=5.0e-5,
        dr_g_s=7.5e-5,
        dot_a_frozen=-0.01,
        Tl_last=300.0,
        Tg_first=320.0,
        Yl_last_full=np.asarray([0.6, 0.4], dtype=np.float64),
        Yg_first_full=np.asarray([0.1, 0.2, 0.3, 0.4], dtype=np.float64),
        Xg_first_full=np.asarray([0.15, 0.15, 0.3, 0.4], dtype=np.float64),
        Ts=310.0,
        Ys_l_full=np.asarray([0.7, 0.3], dtype=np.float64),
        Ys_g_full=np.asarray([0.2, 0.1, 0.3, 0.4], dtype=np.float64),
        Xs_g_full=np.asarray([0.25, 0.05, 0.3, 0.4], dtype=np.float64),
        mpp=0.1,
        rho_s_l=800.0,
        rho_s_g=1.2,
        h_s_l=1.0e5,
        h_s_g=2.0e5,
        h_liq_species_s_full=np.asarray([1.1e5, 0.9e5], dtype=np.float64),
        h_gas_species_s_full=np.asarray([2.1e5, 1.9e5, 1.5e5, 1.4e5], dtype=np.float64),
        k_s_l=0.15,
        k_s_g=0.04,
        D_s_l_full=np.asarray([1.0e-9, 2.0e-9], dtype=np.float64),
        D_s_g_full=np.asarray([1.0e-5, 1.1e-5, 1.2e-5, 1.3e-5], dtype=np.float64),
        dTdr_l_s=1.0,
        dTdr_g_s=2.0,
        dYdr_l_s_full=np.asarray([5.0, -5.0], dtype=np.float64),
        dXdr_g_s_full=np.asarray([1.0, -1.0, 0.0, 0.0], dtype=np.float64),
        J_l_full=np.asarray([0.05, -0.05], dtype=np.float64),
        J_g_full=np.asarray([0.02, -0.01, 0.01, -0.02], dtype=np.float64),
        Vd0_g_full=np.asarray([0.2, -0.1, 0.0, -0.1], dtype=np.float64),
        Vcd_g=-0.03,
        N_l_full=np.asarray([0.8, 0.2], dtype=np.float64),
        N_g_full=np.asarray([0.9, 0.25, 0.5, -0.65], dtype=np.float64),
        q_l_s=10.0,
        q_g_s=20.0,
        q_l_cond=6.0,
        q_g_cond=8.0,
        q_l_species_diff=4.0,
        q_g_species_diff=12.0,
        E_l_s=5.0,
        E_g_s=9.0,
        G_g_if_abs=-1.12e-6,
        Yeq_g_cond_full=np.asarray([0.25, 0.05, 0.3, 0.4], dtype=np.float64),
        gamma_cond_full=np.asarray([1.2, 0.8], dtype=np.float64),
        eps_mass_gas_kinematic=0.0,
    )


def test_interface_residual_single_component_rows_match_layout() -> None:
    maps = _make_species_maps_single()
    layout = _make_layout_single()
    iface_face_pkg = _make_iface_pkg_single()
    iface_mass_pkg = build_interface_mass_residual_package(_make_run_cfg(maps), iface_face_pkg, u_l_if_abs=0.02)
    iface_energy_pkg = build_interface_energy_residual_package(iface_face_pkg)

    result = assemble_interface_residual(
        layout=layout,
        species_maps=maps,
        iface_face_pkg=iface_face_pkg,
        iface_mass_pkg=iface_mass_pkg,
        iface_energy_pkg=iface_energy_pkg,
    )

    assert result.liq_species_rows_global.size == 0
    assert result.Ts_row_global.shape == (1,)
    assert result.Ts_row_global[0] == layout.if_temperature_index
    assert result.gas_species_rows_global.shape == (1,)
    assert result.gas_species_rows_global[0] == layout.if_gas_species_slice.start
    assert result.mpp_row_global.shape == (1,)
    assert result.mpp_row_global[0] == layout.if_mpp_index


def test_interface_residual_multicomponent_rows_and_kinds_match_layout() -> None:
    maps = _make_species_maps_multi()
    layout = _make_layout_multi()
    iface_face_pkg = _make_iface_pkg_multi()
    iface_mass_pkg = build_interface_mass_residual_package(_make_run_cfg(maps), iface_face_pkg, u_l_if_abs=0.02)
    iface_energy_pkg = build_interface_energy_residual_package(iface_face_pkg)

    result = assemble_interface_residual(
        layout=layout,
        species_maps=maps,
        iface_face_pkg=iface_face_pkg,
        iface_mass_pkg=iface_mass_pkg,
        iface_energy_pkg=iface_energy_pkg,
    )

    assert np.array_equal(
        result.rows_global,
        np.asarray(
            [
                layout.if_liq_species_slice.start,
                layout.if_temperature_index,
                layout.if_gas_species_slice.start,
                layout.if_gas_species_slice.start + 1,
                layout.if_gas_species_slice.start + 2,
                layout.if_mpp_index,
            ],
            dtype=np.int64,
        ),
    )
    assert np.array_equal(result.liq_species_rows_global, np.asarray([layout.if_liq_species_slice.start], dtype=np.int64))
    assert np.array_equal(result.Ts_row_global, np.asarray([layout.if_temperature_index], dtype=np.int64))
    assert np.array_equal(
        result.gas_species_rows_global,
        np.arange(layout.if_gas_species_slice.start, layout.if_gas_species_slice.stop, dtype=np.int64),
    )
    assert np.array_equal(result.mpp_row_global, np.asarray([layout.if_mpp_index], dtype=np.int64))
    assert result.gas_species_row_kind.tolist() == ["eq19", "eq19", "eq16"]
    assert np.allclose(result.Ts_value, np.asarray([iface_energy_pkg.energy_residual], dtype=np.float64))
    assert np.allclose(result.liq_species_values, iface_mass_pkg.eq215_values)


def test_interface_residual_rows_follow_frozen_interface_block_order() -> None:
    maps = _make_species_maps_multi()
    layout = _make_layout_multi()
    iface_face_pkg = _make_iface_pkg_multi()
    iface_mass_pkg = build_interface_mass_residual_package(_make_run_cfg(maps), iface_face_pkg, u_l_if_abs=0.02)
    iface_energy_pkg = build_interface_energy_residual_package(iface_face_pkg)

    result = assemble_interface_residual(
        layout=layout,
        species_maps=maps,
        iface_face_pkg=iface_face_pkg,
        iface_mass_pkg=iface_mass_pkg,
        iface_energy_pkg=iface_energy_pkg,
    )

    expected_rows = np.concatenate(
        [
            np.arange(layout.if_liq_species_slice.start, layout.if_liq_species_slice.stop, dtype=np.int64),
            np.asarray([layout.if_temperature_index], dtype=np.int64),
            np.arange(layout.if_gas_species_slice.start, layout.if_gas_species_slice.stop, dtype=np.int64),
            np.asarray([layout.if_mpp_index], dtype=np.int64),
        ]
    )
    assert np.array_equal(result.rows_global, expected_rows)


def test_interface_residual_mpp_row_is_single_and_from_mass_package() -> None:
    maps = _make_species_maps_multi()
    layout = _make_layout_multi()
    iface_face_pkg = _make_iface_pkg_multi()
    iface_mass_pkg = build_interface_mass_residual_package(_make_run_cfg(maps), iface_face_pkg, u_l_if_abs=0.02)
    iface_energy_pkg = build_interface_energy_residual_package(iface_face_pkg)

    result = assemble_interface_residual(
        layout=layout,
        species_maps=maps,
        iface_face_pkg=iface_face_pkg,
        iface_mass_pkg=iface_mass_pkg,
        iface_energy_pkg=iface_energy_pkg,
    )

    assert result.mpp_row_global.shape == (1,)
    assert np.allclose(result.mpp_value, np.asarray([iface_mass_pkg.mpp_residual], dtype=np.float64))
    assert np.count_nonzero(result.rows_global == layout.if_mpp_index) == 1


def test_interface_residual_non_owner_returns_empty() -> None:
    result = assemble_interface_residual(
        layout=_make_layout_multi(),
        species_maps=_make_species_maps_multi(),
        iface_face_pkg=_make_iface_pkg_multi(),
        iface_mass_pkg=build_interface_mass_residual_package(_make_run_cfg(_make_species_maps_multi()), _make_iface_pkg_multi(), u_l_if_abs=0.02),
        iface_energy_pkg=build_interface_energy_residual_package(_make_iface_pkg_multi()),
        owner_active=False,
    )

    assert result.rows_global.size == 0
    assert result.values.size == 0
    assert result.diagnostics == {}


def test_interface_residual_rejects_inconsistent_energy_package() -> None:
    maps = _make_species_maps_multi()
    layout = _make_layout_multi()
    iface_face_pkg = _make_iface_pkg_multi()
    iface_mass_pkg = build_interface_mass_residual_package(_make_run_cfg(maps), iface_face_pkg, u_l_if_abs=0.02)
    iface_energy_pkg = replace(build_interface_energy_residual_package(iface_face_pkg), E_g_s=10.0)

    with pytest.raises(InterfaceResidualAssemblyError, match="E_g_s"):
        assemble_interface_residual(
            layout=layout,
            species_maps=maps,
            iface_face_pkg=iface_face_pkg,
            iface_mass_pkg=iface_mass_pkg,
            iface_energy_pkg=iface_energy_pkg,
        )


def test_interface_residual_rejects_inconsistent_mass_package_against_face_package() -> None:
    maps = _make_species_maps_multi()
    layout = _make_layout_multi()
    iface_face_pkg = _make_iface_pkg_multi()
    iface_mass_pkg = replace(
        build_interface_mass_residual_package(_make_run_cfg(maps), iface_face_pkg, u_l_if_abs=0.02),
        eq216_values=np.asarray([9.9], dtype=np.float64),
    )
    iface_energy_pkg = build_interface_energy_residual_package(iface_face_pkg)

    with pytest.raises(InterfaceResidualAssemblyError, match="eq216_values"):
        assemble_interface_residual(
            layout=layout,
            species_maps=maps,
            iface_face_pkg=iface_face_pkg,
            iface_mass_pkg=iface_mass_pkg,
            iface_energy_pkg=iface_energy_pkg,
        )


def test_interface_residual_rejects_duplicate_reduced_row_mapping() -> None:
    maps = _make_species_maps_multi()
    layout = _make_layout_multi()
    iface_face_pkg = _make_iface_pkg_multi()
    iface_mass_pkg = replace(
        build_interface_mass_residual_package(_make_run_cfg(maps), iface_face_pkg, u_l_if_abs=0.02),
        eq216_gas_full_indices=np.asarray([2], dtype=np.int64),
        eq216_values=np.asarray([0.5], dtype=np.float64),
        eq219_gas_full_indices=np.asarray([0, 0], dtype=np.int64),
        eq219_values=np.asarray([-0.05, -0.05], dtype=np.float64),
    )
    iface_energy_pkg = build_interface_energy_residual_package(iface_face_pkg)

    with pytest.raises(InterfaceResidualAssemblyError, match="filled exactly once"):
        assemble_interface_residual(
            layout=layout,
            species_maps=maps,
            iface_face_pkg=iface_face_pkg,
            iface_mass_pkg=iface_mass_pkg,
            iface_energy_pkg=iface_energy_pkg,
        )
