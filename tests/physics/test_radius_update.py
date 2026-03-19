from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from physics.interface_face import InterfaceFacePackage
from physics.radius_update import (
    RadiusUpdateError,
    build_radius_update_package,
)
from physics.velocity_recovery import PhaseVelocityRecovery, VelocityRecoveryPackage
import physics.radius_update as radius_update


def _make_iface_pkg(*, rho_s_l: float = 800.0, mpp: float = 0.16, dot_a_frozen: float = -0.02) -> InterfaceFacePackage:
    return InterfaceFacePackage(
        r_s=2.0,
        area_s=20.0,
        dr_l_s=0.5,
        dr_g_s=0.5,
        dot_a_frozen=dot_a_frozen,
        Tl_last=301.0,
        Tg_first=320.0,
        Yl_last_full=np.asarray([0.6, 0.4], dtype=np.float64),
        Yg_first_full=np.asarray([0.2, 0.1, 0.3, 0.4], dtype=np.float64),
        Xg_first_full=np.asarray([0.25, 0.05, 0.3, 0.4], dtype=np.float64),
        Ts=310.0,
        Ys_l_full=np.asarray([0.7, 0.3], dtype=np.float64),
        Ys_g_full=np.asarray([0.2, 0.1, 0.3, 0.4], dtype=np.float64),
        Xs_g_full=np.asarray([0.25, 0.05, 0.3, 0.4], dtype=np.float64),
        mpp=mpp,
        rho_s_l=rho_s_l,
        rho_s_g=2.0,
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
        E_l_s=-50.0,
        E_g_s=25.0,
        G_g_if_abs=50.0,
        Yeq_g_cond_full=np.asarray([0.2, 0.1, 0.3, 0.4], dtype=np.float64),
        gamma_cond_full=np.asarray([1.2, 0.8], dtype=np.float64),
        eps_mass_gas_kinematic=0.0,
    )


def _make_phase_recovery() -> PhaseVelocityRecovery:
    return PhaseVelocityRecovery(
        G_face_abs=np.asarray([0.0, 1.0], dtype=np.float64),
        rho_face=np.asarray([1.0, 1.0], dtype=np.float64),
        area_face=np.asarray([1.0, 1.0], dtype=np.float64),
        vc_face=np.asarray([0.0, 0.0], dtype=np.float64),
        u_face_abs=np.asarray([0.0, 0.0], dtype=np.float64),
        u_face_rel=np.asarray([0.0, 0.0], dtype=np.float64),
        current_mass_cell=np.asarray([1.0], dtype=np.float64),
        old_mass_cell=np.asarray([1.0], dtype=np.float64),
    )


def _make_vel_pkg(*, u_l_if_abs: float = -0.01) -> VelocityRecoveryPackage:
    phase = _make_phase_recovery()
    return VelocityRecoveryPackage(
        liquid=phase,
        gas=phase,
        u_l_if_abs=u_l_if_abs,
        u_g_if_abs=0.02,
        G_g_if_abs=50.0,
        diagnostics={"u_l_if_abs": u_l_if_abs},
    )


def test_dot_a_phys_matches_ul_if_plus_mpp_over_rho() -> None:
    iface_pkg = _make_iface_pkg(rho_s_l=800.0, mpp=0.16)
    vel_pkg = _make_vel_pkg(u_l_if_abs=-0.01)

    pkg = build_radius_update_package(iface_pkg, vel_pkg)
    assert np.isclose(pkg.dot_a_phys, -0.01 + 0.16 / 800.0)


def test_eps_dot_a_matches_expected_definition() -> None:
    iface_pkg = _make_iface_pkg(dot_a_frozen=-0.02)
    vel_pkg = _make_vel_pkg(u_l_if_abs=-0.01)

    pkg = build_radius_update_package(iface_pkg, vel_pkg, eps_dot_a_floor=1.0e-12)
    expected = abs(pkg.dot_a_phys + 0.02) / max(abs(pkg.dot_a_phys), 1.0e-12)
    assert np.isclose(pkg.eps_dot_a, expected)


def test_eps_dot_a_uses_floor_when_dot_a_phys_near_zero() -> None:
    iface_pkg = _make_iface_pkg(rho_s_l=2.0, mpp=0.02, dot_a_frozen=1.0e-3)
    vel_pkg = _make_vel_pkg(u_l_if_abs=-0.01)

    pkg = build_radius_update_package(iface_pkg, vel_pkg, eps_dot_a_floor=1.0e-6)
    assert np.isclose(pkg.dot_a_phys, 0.0, atol=1.0e-15)
    assert np.isclose(pkg.eps_dot_a, abs(0.0 - 1.0e-3) / 1.0e-6)


def test_radius_update_consumes_iface_and_velocity_packages_only() -> None:
    assert not hasattr(radius_update, "compute_interface_equilibrium")
    assert not hasattr(radius_update, "build_velocity_recovery_package")
    assert not hasattr(radius_update, "gas_props")
    assert not hasattr(radius_update, "liquid_props")


def test_radius_update_does_not_modify_input_packages() -> None:
    iface_pkg = _make_iface_pkg()
    vel_pkg = _make_vel_pkg()
    before = (iface_pkg.rho_s_l, iface_pkg.mpp, iface_pkg.dot_a_frozen, vel_pkg.u_l_if_abs)

    _ = build_radius_update_package(iface_pkg, vel_pkg)

    after = (iface_pkg.rho_s_l, iface_pkg.mpp, iface_pkg.dot_a_frozen, vel_pkg.u_l_if_abs)
    assert before == after


def test_nonpositive_rho_s_l_raises() -> None:
    with pytest.raises(RadiusUpdateError, match="iface_pkg.rho_s_l must be > 0"):
        build_radius_update_package(_make_iface_pkg(rho_s_l=0.0), _make_vel_pkg())


def test_nonfinite_ul_if_raises() -> None:
    with pytest.raises(RadiusUpdateError, match="vel_pkg.u_l_if_abs must be finite"):
        build_radius_update_package(_make_iface_pkg(), _make_vel_pkg(u_l_if_abs=np.nan))


def test_nonfinite_mpp_raises() -> None:
    with pytest.raises(RadiusUpdateError, match="iface_pkg.mpp must be finite"):
        build_radius_update_package(_make_iface_pkg(mpp=np.nan), _make_vel_pkg())


def test_nonfinite_dot_a_frozen_raises() -> None:
    with pytest.raises(RadiusUpdateError, match="iface_pkg.dot_a_frozen must be finite"):
        build_radius_update_package(_make_iface_pkg(dot_a_frozen=np.nan), _make_vel_pkg())


def test_radius_update_has_no_dt_or_old_radius_dependency() -> None:
    pkg = build_radius_update_package(_make_iface_pkg(), _make_vel_pkg())
    assert not hasattr(pkg, "dt")
    assert not hasattr(pkg, "a_old")
    assert not hasattr(pkg, "state_old")


def test_nonpositive_eps_dot_a_floor_raises() -> None:
    with pytest.raises(RadiusUpdateError, match="eps_dot_a_floor must be > 0"):
        build_radius_update_package(_make_iface_pkg(), _make_vel_pkg(), eps_dot_a_floor=0.0)
