from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from core.types import InterfaceState, Mesh1D, RegionSlices, State
from physics.interface_face import InterfaceFacePackage
from physics.velocity_recovery import (
    OldMassOnCurrentGeometry,
    VelocityRecoveryError,
    build_velocity_recovery_package,
)


def _make_mesh() -> Mesh1D:
    return Mesh1D(
        r_faces=np.asarray([0.0, 1.0, 2.0, 3.0, 4.0], dtype=np.float64),
        r_centers=np.asarray([0.5, 1.5, 2.5, 3.5], dtype=np.float64),
        volumes=np.asarray([2.0, 2.0, 3.0, 3.0], dtype=np.float64),
        face_areas=np.asarray([0.0, 10.0, 20.0, 30.0, 40.0], dtype=np.float64),
        dr=np.asarray([1.0, 1.0, 1.0, 1.0], dtype=np.float64),
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


def _make_state() -> State:
    return State(
        Tl=np.asarray([300.0, 301.0], dtype=np.float64),
        Yl_full=np.asarray([[0.7, 0.3], [0.6, 0.4]], dtype=np.float64),
        Tg=np.asarray([320.0, 321.0], dtype=np.float64),
        Yg_full=np.asarray([[0.2, 0.1, 0.3, 0.4], [0.21, 0.09, 0.3, 0.4]], dtype=np.float64),
        interface=InterfaceState(
            Ts=310.0,
            mpp=0.1,
            Ys_g_full=np.asarray([0.2, 0.1, 0.3, 0.4], dtype=np.float64),
            Ys_l_full=np.asarray([0.7, 0.3], dtype=np.float64),
        ),
        rho_l=np.asarray([10.0, 12.0], dtype=np.float64),
        rho_g=np.asarray([2.0, 3.0], dtype=np.float64),
    )


def _make_iface_pkg(*, G_g_if_abs: float = 50.0) -> InterfaceFacePackage:
    return InterfaceFacePackage(
        r_s=2.0,
        area_s=20.0,
        dr_l_s=0.5,
        dr_g_s=0.5,
        dot_a_frozen=-0.1,
        Tl_last=301.0,
        Tg_first=320.0,
        Yl_last_full=np.asarray([0.6, 0.4], dtype=np.float64),
        Yg_first_full=np.asarray([0.2, 0.1, 0.3, 0.4], dtype=np.float64),
        Xg_first_full=np.asarray([0.25, 0.05, 0.3, 0.4], dtype=np.float64),
        Ts=310.0,
        Ys_l_full=np.asarray([0.7, 0.3], dtype=np.float64),
        Ys_g_full=np.asarray([0.2, 0.1, 0.3, 0.4], dtype=np.float64),
        Xs_g_full=np.asarray([0.25, 0.05, 0.3, 0.4], dtype=np.float64),
        mpp=0.1,
        rho_s_l=12.0,
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
        G_g_if_abs=G_g_if_abs,
        Yeq_g_cond_full=np.asarray([0.2, 0.1, 0.3, 0.4], dtype=np.float64),
        gamma_cond_full=np.asarray([1.2, 0.8], dtype=np.float64),
        eps_mass_gas_kinematic=0.0,
    )


def _build_pkg(
    *,
    old_mass: Any | None = None,
    vc_face_liq: np.ndarray | None = None,
    vc_face_gas: np.ndarray | None = None,
    dt: float = 2.0,
    iface_G: float = 50.0,
) -> Any:
    mesh = _make_mesh()
    state = _make_state()
    if old_mass is None:
        old_mass = OldMassOnCurrentGeometry(
            mass_cell_liq=np.asarray([18.0, 23.0], dtype=np.float64),
            mass_cell_gas=np.asarray([5.0, 11.0], dtype=np.float64),
        )
    if vc_face_liq is None:
        vc_face_liq = np.asarray([0.0, 0.1, 0.2], dtype=np.float64)
    if vc_face_gas is None:
        vc_face_gas = np.asarray([-0.1, -0.2, -0.3], dtype=np.float64)
    return build_velocity_recovery_package(
        mesh=mesh,
        state=state,
        old_mass_on_current_geometry=old_mass,
        iface_pkg=_make_iface_pkg(G_g_if_abs=iface_G),
        vc_face_liq=vc_face_liq,
        vc_face_gas=vc_face_gas,
        dt=dt,
    )


def test_liquid_center_face_mass_flux_is_zero() -> None:
    pkg = _build_pkg()
    assert np.isclose(pkg.liquid.G_face_abs[0], 0.0)


def test_liquid_interface_velocity_matches_last_face() -> None:
    pkg = _build_pkg()
    assert np.isclose(pkg.u_l_if_abs, pkg.liquid.u_face_abs[-1])


def test_gas_interface_face_starts_from_interface_package_mass_flux() -> None:
    pkg = _build_pkg(iface_G=47.0)
    assert np.isclose(pkg.gas.G_face_abs[0], 47.0)
    assert np.isclose(pkg.G_g_if_abs, 47.0)


def test_gas_interface_velocity_is_face_zero_velocity() -> None:
    pkg = _build_pkg()
    assert np.isclose(pkg.u_g_if_abs, pkg.gas.u_face_abs[0])


def test_liquid_sweep_satisfies_G_right_equals_G_left_plus_C_minus_S() -> None:
    pkg = _build_pkg()
    for n in range(pkg.liquid.current_mass_cell.shape[0]):
        f_left = n
        f_right = n + 1
        C_n = (
            pkg.liquid.rho_face[f_right] * pkg.liquid.vc_face[f_right] * pkg.liquid.area_face[f_right]
            - pkg.liquid.rho_face[f_left] * pkg.liquid.vc_face[f_left] * pkg.liquid.area_face[f_left]
        )
        S_n = (pkg.liquid.current_mass_cell[n] - pkg.liquid.old_mass_cell[n]) / 2.0
        assert np.isclose(pkg.liquid.G_face_abs[f_right], pkg.liquid.G_face_abs[f_left] + C_n - S_n)


def test_gas_sweep_satisfies_G_right_equals_G_left_plus_C_minus_S() -> None:
    pkg = _build_pkg()
    for n in range(pkg.gas.current_mass_cell.shape[0]):
        f_left = n
        f_right = n + 1
        C_n = (
            pkg.gas.rho_face[f_right] * pkg.gas.vc_face[f_right] * pkg.gas.area_face[f_right]
            - pkg.gas.rho_face[f_left] * pkg.gas.vc_face[f_left] * pkg.gas.area_face[f_left]
        )
        S_n = (pkg.gas.current_mass_cell[n] - pkg.gas.old_mass_cell[n]) / 2.0
        assert np.isclose(pkg.gas.G_face_abs[f_right], pkg.gas.G_face_abs[f_left] + C_n - S_n)


def test_velocity_recovery_uses_old_mass_on_current_geometry() -> None:
    pkg_a = _build_pkg(old_mass=OldMassOnCurrentGeometry(
        mass_cell_liq=np.asarray([18.0, 23.0], dtype=np.float64),
        mass_cell_gas=np.asarray([5.0, 11.0], dtype=np.float64),
    ))
    pkg_b = _build_pkg(old_mass=OldMassOnCurrentGeometry(
        mass_cell_liq=np.asarray([1.0, 2.0], dtype=np.float64),
        mass_cell_gas=np.asarray([30.0, 40.0], dtype=np.float64),
    ))

    assert not np.allclose(pkg_a.liquid.G_face_abs, pkg_b.liquid.G_face_abs)
    assert not np.allclose(pkg_a.gas.G_face_abs, pkg_b.gas.G_face_abs)


def test_relative_velocity_is_u_abs_minus_vc_for_liquid() -> None:
    pkg = _build_pkg()
    assert np.allclose(pkg.liquid.u_face_rel, pkg.liquid.u_face_abs - pkg.liquid.vc_face)


def test_relative_velocity_is_u_abs_minus_vc_for_gas() -> None:
    pkg = _build_pkg()
    assert np.allclose(pkg.gas.u_face_rel, pkg.gas.u_face_abs - pkg.gas.vc_face)


def test_gas_farfield_velocity_is_recovered_not_prescribed() -> None:
    pkg = _build_pkg()
    n = pkg.gas.current_mass_cell.shape[0] - 1
    f_left = n
    f_right = n + 1
    C_n = (
        pkg.gas.rho_face[f_right] * pkg.gas.vc_face[f_right] * pkg.gas.area_face[f_right]
        - pkg.gas.rho_face[f_left] * pkg.gas.vc_face[f_left] * pkg.gas.area_face[f_left]
    )
    S_n = (pkg.gas.current_mass_cell[n] - pkg.gas.old_mass_cell[n]) / 2.0
    expected_G_far = pkg.gas.G_face_abs[f_left] + C_n - S_n
    assert np.isclose(pkg.gas.G_face_abs[f_right], expected_G_far)


def test_liquid_face_density_rule() -> None:
    pkg = _build_pkg()
    assert np.allclose(pkg.liquid.rho_face, np.asarray([10.0, 11.0, 12.0], dtype=np.float64))


def test_gas_face_density_rule() -> None:
    pkg = _build_pkg()
    assert np.allclose(pkg.gas.rho_face, np.asarray([2.0, 2.5, 3.0], dtype=np.float64))


def test_nonpositive_dt_raises() -> None:
    with pytest.raises(VelocityRecoveryError, match="dt must be > 0"):
        _build_pkg(dt=0.0)


def test_nonfinite_old_mass_raises() -> None:
    with pytest.raises(VelocityRecoveryError, match="must contain only finite values"):
        _build_pkg(old_mass=OldMassOnCurrentGeometry(
            mass_cell_liq=np.asarray([18.0, np.nan], dtype=np.float64),
            mass_cell_gas=np.asarray([5.0, 11.0], dtype=np.float64),
        ))


def test_nonpositive_face_area_raises() -> None:
    mesh = _make_mesh()
    mesh = Mesh1D(
        r_faces=mesh.r_faces.copy(),
        r_centers=mesh.r_centers.copy(),
        volumes=mesh.volumes.copy(),
        face_areas=np.asarray([0.0, 10.0, 0.0, 30.0, 40.0], dtype=np.float64),
        dr=mesh.dr.copy(),
        region_slices=mesh.region_slices,
        interface_face_index=mesh.interface_face_index,
        interface_cell_liq=mesh.interface_cell_liq,
        interface_cell_gas=mesh.interface_cell_gas,
    )
    with pytest.raises(VelocityRecoveryError, match="non-center liquid face areas must be strictly positive"):
        build_velocity_recovery_package(
            mesh=mesh,
            state=_make_state(),
            old_mass_on_current_geometry=OldMassOnCurrentGeometry(
                mass_cell_liq=np.asarray([18.0, 23.0], dtype=np.float64),
                mass_cell_gas=np.asarray([5.0, 11.0], dtype=np.float64),
            ),
            iface_pkg=_make_iface_pkg(),
            vc_face_liq=np.asarray([0.0, 0.1, 0.2], dtype=np.float64),
            vc_face_gas=np.asarray([-0.1, -0.2, -0.3], dtype=np.float64),
            dt=2.0,
        )


def test_nonpositive_cell_volume_raises() -> None:
    mesh = _make_mesh()
    object.__setattr__(mesh, "volumes", np.asarray([2.0, 0.0, 3.0, 3.0], dtype=np.float64))
    with pytest.raises(VelocityRecoveryError, match="liquid cell volumes must be strictly positive"):
        build_velocity_recovery_package(
            mesh=mesh,
            state=_make_state(),
            old_mass_on_current_geometry=OldMassOnCurrentGeometry(
                mass_cell_liq=np.asarray([18.0, 23.0], dtype=np.float64),
                mass_cell_gas=np.asarray([5.0, 11.0], dtype=np.float64),
            ),
            iface_pkg=_make_iface_pkg(),
            vc_face_liq=np.asarray([0.0, 0.1, 0.2], dtype=np.float64),
            vc_face_gas=np.asarray([-0.1, -0.2, -0.3], dtype=np.float64),
            dt=2.0,
        )


def test_nonfinite_interface_mass_flux_raises() -> None:
    with pytest.raises(VelocityRecoveryError, match="iface_pkg.G_g_if_abs must be finite"):
        _build_pkg(iface_G=np.nan)


def test_density_shape_mismatch_raises() -> None:
    mesh = Mesh1D(
        r_faces=np.asarray([0.0, 1.0, 2.0, 3.0, 4.0], dtype=np.float64),
        r_centers=np.asarray([0.5, 1.5, 2.5, 3.5], dtype=np.float64),
        volumes=np.asarray([2.0, 2.0, 3.0, 3.0], dtype=np.float64),
        face_areas=np.asarray([0.0, 10.0, 20.0, 30.0, 40.0], dtype=np.float64),
        dr=np.asarray([1.0, 1.0, 1.0, 1.0], dtype=np.float64),
        region_slices=RegionSlices(
            liq=slice(0, 1),
            gas_near=slice(1, 2),
            gas_far=slice(2, 4),
            gas_all=slice(1, 4),
        ),
        interface_face_index=1,
        interface_cell_liq=0,
        interface_cell_gas=1,
    )
    with pytest.raises(VelocityRecoveryError, match="state.rho_l must have length 1"):
        build_velocity_recovery_package(
            mesh=mesh,
            state=_make_state(),
            old_mass_on_current_geometry=OldMassOnCurrentGeometry(
                mass_cell_liq=np.asarray([18.0], dtype=np.float64),
                mass_cell_gas=np.asarray([5.0, 11.0, 13.0], dtype=np.float64),
            ),
            iface_pkg=_make_iface_pkg(),
            vc_face_liq=np.asarray([0.0, 0.2], dtype=np.float64),
            vc_face_gas=np.asarray([-0.1, -0.2, -0.3, -0.4], dtype=np.float64),
            dt=2.0,
        )


def test_supports_old_mass_namespace_with_explicit_fields() -> None:
    old_mass = SimpleNamespace(
        mass_cell_liq=np.asarray([18.0, 23.0], dtype=np.float64),
        mass_cell_gas=np.asarray([5.0, 11.0], dtype=np.float64),
    )
    pkg = _build_pkg(old_mass=old_mass)
    assert np.allclose(pkg.liquid.old_mass_cell, old_mass.mass_cell_liq)
    assert np.allclose(pkg.gas.old_mass_cell, old_mass.mass_cell_gas)
