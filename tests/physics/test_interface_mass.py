from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from core.types import SpeciesMaps
from physics.interface_face import InterfaceFacePackage
from physics.interface_mass import (
    InterfaceMassError,
    build_interface_mass_residual_package,
)
import physics.interface_mass as interface_mass


def _make_species_maps(*, liq_to_gas_full: np.ndarray | None = None) -> SpeciesMaps:
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
        liq_full_to_gas_full=np.asarray([0, 1] if liq_to_gas_full is None else liq_to_gas_full, dtype=np.int64),
    )


def _make_run_cfg(species_maps: SpeciesMaps) -> SimpleNamespace:
    return SimpleNamespace(species_maps=species_maps)


def _make_iface_pkg(
    *,
    Yeq_g_cond_full: np.ndarray | None = None,
    G_g_if_abs: float | None = None,
) -> InterfaceFacePackage:
    area_s = float(4.0 * np.pi * 1.0e-6)
    rho_s_g = 1.2
    dot_a_frozen = -0.01
    mpp = 0.1
    if G_g_if_abs is None:
        G_g_if_abs = (rho_s_g * dot_a_frozen - mpp) * area_s

    return InterfaceFacePackage(
        r_s=1.0e-3,
        area_s=area_s,
        dr_l_s=5.0e-5,
        dr_g_s=7.5e-5,
        dot_a_frozen=dot_a_frozen,
        Tl_last=300.0,
        Tg_first=320.0,
        Yl_last_full=np.asarray([0.6, 0.4], dtype=np.float64),
        Yg_first_full=np.asarray([0.1, 0.2, 0.3, 0.4], dtype=np.float64),
        Xg_first_full=np.asarray([0.15, 0.15, 0.3, 0.4], dtype=np.float64),
        Ts=310.0,
        Ys_l_full=np.asarray([0.7, 0.3], dtype=np.float64),
        Ys_g_full=np.asarray([0.2, 0.1, 0.3, 0.4], dtype=np.float64),
        Xs_g_full=np.asarray([0.25, 0.05, 0.3, 0.4], dtype=np.float64),
        mpp=mpp,
        rho_s_l=800.0,
        rho_s_g=rho_s_g,
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
        E_l_s=0.0,
        E_g_s=0.0,
        G_g_if_abs=float(G_g_if_abs),
        Yeq_g_cond_full=None if Yeq_g_cond_full is None else np.asarray(Yeq_g_cond_full, dtype=np.float64),
        gamma_cond_full=np.asarray([1.2, 0.8], dtype=np.float64),
        eps_mass_gas_kinematic=0.0,
    )


def test_interface_mass_consumes_interface_face_package_only() -> None:
    assert not hasattr(interface_mass, "compute_interface_equilibrium")
    assert not hasattr(interface_mass, "compute_total_energy_flux_density")


def test_eq215_uses_total_species_flux_difference() -> None:
    pkg = build_interface_mass_residual_package(
        _make_run_cfg(_make_species_maps()),
        _make_iface_pkg(Yeq_g_cond_full=np.asarray([0.25, 0.05, 0.3, 0.4])),
        u_l_if_abs=0.02,
    )

    assert np.array_equal(pkg.eq215_liq_full_indices, np.asarray([0], dtype=np.int64))
    assert np.array_equal(pkg.eq215_gas_full_indices, np.asarray([0], dtype=np.int64))
    assert np.allclose(pkg.eq215_values, np.asarray([0.1], dtype=np.float64))


def test_eq215_excludes_liquid_closure_species_when_present() -> None:
    pkg = build_interface_mass_residual_package(
        _make_run_cfg(_make_species_maps()),
        _make_iface_pkg(Yeq_g_cond_full=np.asarray([0.25, 0.05, 0.3, 0.4])),
        u_l_if_abs=0.02,
    )

    assert 1 not in pkg.eq215_liq_full_indices.tolist()


def test_eq216_uses_noncondensable_gas_flux_only() -> None:
    pkg = build_interface_mass_residual_package(
        _make_run_cfg(_make_species_maps()),
        _make_iface_pkg(Yeq_g_cond_full=np.asarray([0.25, 0.05, 0.3, 0.4])),
        u_l_if_abs=0.02,
    )

    assert np.array_equal(pkg.eq216_gas_full_indices, np.asarray([2], dtype=np.int64))
    assert np.allclose(pkg.eq216_values, np.asarray([0.5], dtype=np.float64))


def test_eq216_excludes_gas_closure_species() -> None:
    pkg = build_interface_mass_residual_package(
        _make_run_cfg(_make_species_maps()),
        _make_iface_pkg(Yeq_g_cond_full=np.asarray([0.25, 0.05, 0.3, 0.4])),
        u_l_if_abs=0.02,
    )

    assert 3 not in pkg.eq216_gas_full_indices.tolist()


def test_eq219_uses_interface_gas_minus_equilibrium_gas() -> None:
    pkg = build_interface_mass_residual_package(
        _make_run_cfg(_make_species_maps()),
        _make_iface_pkg(Yeq_g_cond_full=np.asarray([0.25, 0.05, 0.3, 0.4])),
        u_l_if_abs=0.02,
    )

    assert np.array_equal(pkg.eq219_gas_full_indices, np.asarray([0, 1], dtype=np.int64))
    assert np.allclose(pkg.eq219_values, np.asarray([-0.05, 0.05], dtype=np.float64))


def test_eq219_uses_full_gas_equilibrium_vector() -> None:
    pkg = build_interface_mass_residual_package(
        _make_run_cfg(_make_species_maps()),
        _make_iface_pkg(Yeq_g_cond_full=np.asarray([0.21, 0.11, 9.9, 8.8])),
        u_l_if_abs=0.02,
    )

    assert np.allclose(pkg.eq219_values, np.asarray([-0.01, -0.01], dtype=np.float64))


def test_eq219_excludes_gas_closure_species() -> None:
    pkg = build_interface_mass_residual_package(
        _make_run_cfg(_make_species_maps()),
        _make_iface_pkg(Yeq_g_cond_full=np.asarray([0.25, 0.05, 0.3, 0.4])),
        u_l_if_abs=0.02,
    )

    assert 3 not in pkg.eq219_gas_full_indices.tolist()


def test_mpp_residual_uses_liquid_side_only() -> None:
    pkg = build_interface_mass_residual_package(
        _make_run_cfg(_make_species_maps()),
        _make_iface_pkg(Yeq_g_cond_full=np.asarray([0.25, 0.05, 0.3, 0.4])),
        u_l_if_abs=0.02,
    )

    assert np.isclose(pkg.mpp_residual, 24.1)


def test_interface_mass_does_not_create_second_gas_mpp_row() -> None:
    pkg = build_interface_mass_residual_package(
        _make_run_cfg(_make_species_maps()),
        _make_iface_pkg(Yeq_g_cond_full=np.asarray([0.25, 0.05, 0.3, 0.4])),
        u_l_if_abs=0.02,
    )

    assert hasattr(pkg, "mpp_residual")
    assert not hasattr(pkg, "gas_mpp_residual")


def test_gas_eq18_diag_optional() -> None:
    maps = _make_species_maps()
    iface_pkg = _make_iface_pkg(Yeq_g_cond_full=np.asarray([0.25, 0.05, 0.3, 0.4]))

    pkg_without = build_interface_mass_residual_package(_make_run_cfg(maps), iface_pkg, u_l_if_abs=0.02)
    assert pkg_without.gas_eq18_diag is None

    u_g_if_abs = iface_pkg.dot_a_frozen - iface_pkg.mpp / iface_pkg.rho_s_g
    pkg_with = build_interface_mass_residual_package(
        _make_run_cfg(maps),
        iface_pkg,
        u_l_if_abs=0.02,
        u_g_if_abs=u_g_if_abs,
    )
    assert np.isclose(pkg_with.gas_eq18_diag, 0.0, atol=1.0e-12)


def test_gas_eq18_from_G_diag_consistent_with_G_g_if_abs() -> None:
    pkg = build_interface_mass_residual_package(
        _make_run_cfg(_make_species_maps()),
        _make_iface_pkg(Yeq_g_cond_full=np.asarray([0.25, 0.05, 0.3, 0.4])),
        u_l_if_abs=0.02,
    )

    assert np.isclose(pkg.gas_eq18_from_G_diag, 0.0, atol=1.0e-12)


def test_missing_equilibrium_vector_raises_for_eq219() -> None:
    with pytest.raises(InterfaceMassError, match="Eq.\\(2\\.19\\) requires iface_pkg.Yeq_g_cond_full"):
        build_interface_mass_residual_package(
            _make_run_cfg(_make_species_maps()),
            _make_iface_pkg(Yeq_g_cond_full=None),
            u_l_if_abs=0.02,
        )


def test_nonfinite_ul_if_raises() -> None:
    with pytest.raises(InterfaceMassError, match="u_l_if_abs must be finite"):
        build_interface_mass_residual_package(
            _make_run_cfg(_make_species_maps()),
            _make_iface_pkg(Yeq_g_cond_full=np.asarray([0.25, 0.05, 0.3, 0.4])),
            u_l_if_abs=np.nan,
        )


def test_species_index_sets_are_disjoint_and_valid() -> None:
    pkg = build_interface_mass_residual_package(
        _make_run_cfg(_make_species_maps()),
        _make_iface_pkg(Yeq_g_cond_full=np.asarray([0.25, 0.05, 0.3, 0.4])),
        u_l_if_abs=0.02,
    )

    assert set(pkg.eq216_gas_full_indices.tolist()).isdisjoint(set(pkg.eq219_gas_full_indices.tolist()))
    assert np.all(pkg.eq215_liq_full_indices >= 0)
    assert np.all(pkg.eq215_gas_full_indices >= 0)
    assert np.all(pkg.eq216_gas_full_indices >= 0)
    assert np.all(pkg.eq219_gas_full_indices >= 0)
