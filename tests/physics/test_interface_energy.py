from __future__ import annotations

import numpy as np
import pytest

from physics.interface_energy import (
    InterfaceEnergyError,
    build_interface_energy_residual_package,
)
from physics.interface_face import InterfaceFacePackage
import physics.interface_energy as interface_energy


def _make_iface_pkg() -> InterfaceFacePackage:
    return InterfaceFacePackage(
        r_s=1.0e-3,
        area_s=1.0e-5,
        dr_l_s=5.0e-5,
        dr_g_s=7.0e-5,
        dot_a_frozen=-0.01,
        Tl_last=300.0,
        Tg_first=320.0,
        Yl_last_full=np.asarray([0.6, 0.4], dtype=np.float64),
        Yg_first_full=np.asarray([0.1, 0.2, 0.3, 0.4], dtype=np.float64),
        Xg_first_full=np.asarray([0.12, 0.18, 0.30, 0.40], dtype=np.float64),
        Ts=310.0,
        Ys_l_full=np.asarray([0.7, 0.3], dtype=np.float64),
        Ys_g_full=np.asarray([0.2, 0.1, 0.3, 0.4], dtype=np.float64),
        Xs_g_full=np.asarray([0.25, 0.05, 0.30, 0.40], dtype=np.float64),
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
        E_l_s=-50.0,
        E_g_s=25.0,
        G_g_if_abs=-1.0e-6,
        Yeq_g_cond_full=np.asarray([0.2, 0.1, 0.3, 0.4], dtype=np.float64),
        gamma_cond_full=np.asarray([1.2, 0.8], dtype=np.float64),
        eps_mass_gas_kinematic=0.0,
    )


def test_interface_energy_residual_is_Eg_minus_El() -> None:
    pkg = build_interface_energy_residual_package(_make_iface_pkg())
    assert np.isclose(pkg.energy_residual, 75.0)


def test_interface_energy_consumes_interface_face_package_only() -> None:
    assert not hasattr(interface_energy, "compute_total_energy_flux_density")
    assert not hasattr(interface_energy, "compute_interface_equilibrium")


def test_interface_energy_package_exposes_q_and_E_splits() -> None:
    pkg = build_interface_energy_residual_package(_make_iface_pkg())

    assert np.isclose(pkg.q_l_s, 10.0)
    assert np.isclose(pkg.q_g_s, 20.0)
    assert np.isclose(pkg.E_l_s, -50.0)
    assert np.isclose(pkg.E_g_s, 25.0)
    assert np.isclose(pkg.q_l_cond, 6.0)
    assert np.isclose(pkg.q_g_cond, 8.0)
    assert np.isclose(pkg.q_l_species_diff, 4.0)
    assert np.isclose(pkg.q_g_species_diff, 12.0)


def test_interface_energy_preserves_species_diffusion_heat_contribution() -> None:
    pkg = build_interface_energy_residual_package(_make_iface_pkg())

    assert np.isclose(pkg.q_l_species_diff, 4.0)
    assert np.isclose(pkg.q_g_species_diff, 12.0)
    assert np.isclose(pkg.diagnostics["q_l_species_diff"], 4.0)
    assert np.isclose(pkg.diagnostics["q_g_species_diff"], 12.0)


@pytest.mark.parametrize("field_name", ["E_l_s", "E_g_s", "q_l_s", "q_g_s", "q_l_cond", "q_g_cond", "q_l_species_diff", "q_g_species_diff"])
def test_nonfinite_E_or_q_raises(field_name: str) -> None:
    iface_pkg = _make_iface_pkg()
    iface_pkg = InterfaceFacePackage(**{**iface_pkg.__dict__, field_name: np.nan})

    with pytest.raises(InterfaceEnergyError, match=f"iface_pkg\\.{field_name} must be finite"):
        build_interface_energy_residual_package(iface_pkg)
