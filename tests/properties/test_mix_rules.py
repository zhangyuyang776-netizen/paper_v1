from __future__ import annotations

import numpy as np
import pytest

from properties.mix_rules import (
    MixRulesModelError,
    MixRulesValidationError,
    log_mole_weighted_average,
    mass_to_mole_fractions,
    mass_weighted_sum,
    mixture_liquid_conductivity,
    mixture_liquid_cp,
    mixture_liquid_density,
    mixture_liquid_diffusivity,
    mixture_liquid_enthalpy,
    mixture_liquid_viscosity,
    mixture_molecular_weight_from_mass_fractions,
    mixture_molecular_weight_from_mole_fractions,
    mole_to_mass_fractions,
    mole_weighted_sum,
)


def test_mass_to_mole_to_mass_roundtrip() -> None:
    Y = np.array([0.25, 0.75], dtype=np.float64)
    mw = np.array([0.02, 0.04], dtype=np.float64)
    X = mass_to_mole_fractions(Y, mw)
    Y_back = mole_to_mass_fractions(X, mw)
    assert np.allclose(Y_back, Y)


def test_mole_to_mass_to_mole_roundtrip() -> None:
    X = np.array([0.6, 0.4], dtype=np.float64)
    mw = np.array([0.018, 0.046], dtype=np.float64)
    Y = mole_to_mass_fractions(X, mw)
    X_back = mass_to_mole_fractions(Y, mw)
    assert np.allclose(X_back, X)


def test_single_component_naturally_degenerates() -> None:
    Y = np.array([1.0], dtype=np.float64)
    X = np.array([1.0], dtype=np.float64)
    mw = np.array([0.046], dtype=np.float64)
    rho = np.array([789.0], dtype=np.float64)
    k = np.array([0.17], dtype=np.float64)
    mu = np.array([1.3e-3], dtype=np.float64)
    cp = np.array([2400.0], dtype=np.float64)
    h = np.array([1.2e5], dtype=np.float64)

    assert np.allclose(mass_to_mole_fractions(Y, mw), np.array([1.0]))
    assert np.allclose(mole_to_mass_fractions(X, mw), np.array([1.0]))
    assert mixture_molecular_weight_from_mass_fractions(Y, mw) == pytest.approx(0.046)
    assert mixture_molecular_weight_from_mole_fractions(X, mw) == pytest.approx(0.046)
    assert mixture_liquid_density(Y, X, rho, model="merino_x_sqrt_rho") == pytest.approx(rho[0])
    assert mixture_liquid_cp(Y, cp, model="mass_weighted") == pytest.approx(cp[0])
    assert mixture_liquid_enthalpy(Y, h, model="mass_weighted") == pytest.approx(h[0])
    assert mixture_liquid_conductivity(Y, X, k, model="filippov") == pytest.approx(k[0])
    assert mixture_liquid_viscosity(Y, X, mu, model="grunberg_nissan") == pytest.approx(mu[0])


def test_length_mismatch_raises() -> None:
    with pytest.raises(MixRulesValidationError, match="length"):
        mass_to_mole_fractions(np.array([0.5, 0.5]), np.array([0.018]))


def test_negative_mass_fraction_raises() -> None:
    with pytest.raises(MixRulesValidationError, match="non-negative"):
        mass_to_mole_fractions(np.array([-0.1, 1.1]), np.array([0.018, 0.046]))


def test_mass_fraction_sum_mismatch_raises() -> None:
    with pytest.raises(MixRulesValidationError, match="sum to 1"):
        mass_to_mole_fractions(np.array([0.2, 0.2]), np.array([0.018, 0.046]))


def test_density_merino_x_sqrt_rho() -> None:
    Y = np.array([0.25, 0.75], dtype=np.float64)
    X = np.array([0.4, 0.6], dtype=np.float64)
    rho = np.array([1000.0, 800.0], dtype=np.float64)
    rho_mix = mixture_liquid_density(Y, X, rho, model="merino_x_sqrt_rho")
    expected = float((np.sum(X * np.sqrt(rho))) ** 2)
    assert rho_mix == pytest.approx(expected)


def test_cp_mass_weighted() -> None:
    Y = np.array([0.25, 0.75], dtype=np.float64)
    cp = np.array([1000.0, 2000.0], dtype=np.float64)
    assert mixture_liquid_cp(Y, cp, model="mass_weighted") == pytest.approx(1750.0)


def test_enthalpy_mass_weighted() -> None:
    Y = np.array([0.25, 0.75], dtype=np.float64)
    h = np.array([1.0e5, 2.0e5], dtype=np.float64)
    assert mixture_liquid_enthalpy(Y, h, model="mass_weighted") == pytest.approx(1.75e5)


def test_conductivity_filippov() -> None:
    Y = np.array([0.3, 0.7], dtype=np.float64)
    X = np.array([0.5, 0.5], dtype=np.float64)
    k = np.array([0.10, 0.20], dtype=np.float64)
    expected = Y[0] * (k[0] - 0.72 * Y[1] * abs(k[0] - k[1])) + Y[1] * k[1]
    assert mixture_liquid_conductivity(Y, X, k, model="filippov") == pytest.approx(expected)


def test_conductivity_unsupported_model_raises() -> None:
    with pytest.raises(MixRulesModelError, match="Unsupported liquid conductivity mixture model"):
        mixture_liquid_conductivity(
            np.array([0.5, 0.5]),
            np.array([0.5, 0.5]),
            np.array([0.10, 0.20]),
            model="unknown",
        )


def test_viscosity_grunberg_nissan() -> None:
    Y = np.array([0.4, 0.6], dtype=np.float64)
    X = np.array([0.25, 0.75], dtype=np.float64)
    mu = np.array([1.0e-3, 4.0e-3], dtype=np.float64)
    expected = float(np.exp(np.sum(X * np.log(mu))))
    assert mixture_liquid_viscosity(Y, X, mu, model="grunberg_nissan") == pytest.approx(expected)


def test_viscosity_nonpositive_component_raises() -> None:
    with pytest.raises(MixRulesValidationError, match="strictly positive"):
        mixture_liquid_viscosity(
            np.array([0.5, 0.5]),
            np.array([0.5, 0.5]),
            np.array([1.0e-3, 0.0]),
            model="grunberg_nissan",
        )


def test_mixture_liquid_diffusivity_wilke_chang() -> None:
    Y = np.array([0.4, 0.6], dtype=np.float64)
    X = np.array([0.3, 0.7], dtype=np.float64)
    mw = np.array([0.04607, 0.01801528], dtype=np.float64)
    phi = np.array([1.5, 2.6], dtype=np.float64)
    V = np.array([58.0, 18.0], dtype=np.float64)
    mu_mix = 1.2e-3
    T = 320.0

    D = mixture_liquid_diffusivity(
        Y,
        X,
        model="wilke_chang",
        T=T,
        mu_mix=mu_mix,
        molecular_weights=mw,
        association_factors=phi,
        molar_volumes=V,
    )

    expected = np.array(
        [
            1.173e-16 * np.sqrt(X[1] * phi[1] * (mw[1] * 1000.0)) * T / (mu_mix * ((V[0] / 1000.0) ** 0.6)),
            1.173e-16 * np.sqrt(X[0] * phi[0] * (mw[0] * 1000.0)) * T / (mu_mix * ((V[1] / 1000.0) ** 0.6)),
        ],
        dtype=np.float64,
    )
    assert np.allclose(D, expected)


def test_mixture_liquid_diffusivity_wilke_chang_missing_inputs_raise() -> None:
    with pytest.raises(MixRulesValidationError, match="mu_mix"):
        mixture_liquid_diffusivity(
            np.array([0.5, 0.5]),
            np.array([0.5, 0.5]),
            model="wilke_chang",
            T=300.0,
            molecular_weights=np.array([0.02, 0.03]),
            association_factors=np.array([1.0, 1.0]),
            molar_volumes=np.array([10.0, 12.0]),
        )


def test_mixture_liquid_diffusivity_invalid_model_raises() -> None:
    with pytest.raises(MixRulesModelError, match="Unsupported liquid diffusivity mixture model"):
        mixture_liquid_diffusivity(
            np.array([0.5, 0.5]),
            np.array([0.5, 0.5]),
            model="unsupported",
            T=300.0,
            mu_mix=1.0e-3,
            molecular_weights=np.array([0.02, 0.03]),
            association_factors=np.array([1.0, 1.0]),
            molar_volumes=np.array([10.0, 12.0]),
        )


def test_basic_weighted_helpers() -> None:
    Y = np.array([0.2, 0.8], dtype=np.float64)
    X = np.array([0.25, 0.75], dtype=np.float64)
    values = np.array([10.0, 20.0], dtype=np.float64)

    assert mass_weighted_sum(Y, values) == pytest.approx(18.0)
    assert mole_weighted_sum(X, values) == pytest.approx(17.5)
    assert log_mole_weighted_average(X, np.array([1.0e-3, 4.0e-3])) == pytest.approx(
        float(np.exp(np.sum(X * np.log(np.array([1.0e-3, 4.0e-3])))))
    )
