from __future__ import annotations

from typing import Any

import numpy as np


class MixRulesError(Exception):
    """Base error for liquid mixture-rule helpers."""


class MixRulesValidationError(MixRulesError):
    """Raised when composition or pure-property inputs are structurally invalid."""


class MixRulesModelError(MixRulesError):
    """Raised when a requested mixture model is not supported."""


def _as_1d_float_array(name: str, value: Any) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim != 1:
        raise MixRulesValidationError(f"{name} must be a one-dimensional float array")
    if arr.size == 0:
        raise MixRulesValidationError(f"{name} must be non-empty")
    if not np.all(np.isfinite(arr)):
        raise MixRulesValidationError(f"{name} must contain only finite values")
    return arr


def _validate_same_size(*arrays: np.ndarray, names: tuple[str, ...]) -> None:
    if len(arrays) != len(names):
        raise ValueError("names length must match the number of arrays")
    size = arrays[0].size
    for name, arr in zip(names, arrays, strict=True):
        if arr.size != size:
            raise MixRulesValidationError(f"{name} must have length {size}")


def _validate_mass_fractions(Y: np.ndarray, *, atol: float = 1.0e-12) -> None:
    if np.any(Y < 0.0):
        raise MixRulesValidationError("mass fractions must be non-negative")
    if not np.isclose(float(np.sum(Y)), 1.0, rtol=0.0, atol=atol):
        raise MixRulesValidationError("mass fractions must sum to 1 within tolerance")


def _validate_mole_fractions(X: np.ndarray, *, atol: float = 1.0e-12) -> None:
    if np.any(X < 0.0):
        raise MixRulesValidationError("mole fractions must be non-negative")
    if not np.isclose(float(np.sum(X)), 1.0, rtol=0.0, atol=atol):
        raise MixRulesValidationError("mole fractions must sum to 1 within tolerance")


def _validate_molecular_weights(mw: np.ndarray) -> None:
    if np.any(mw <= 0.0):
        raise MixRulesValidationError("molecular weights must be strictly positive")


def _validate_positive_values(name: str, values: np.ndarray) -> None:
    if np.any(values <= 0.0):
        raise MixRulesValidationError(f"{name} must be strictly positive")


def _filippov_conductivity(Y: np.ndarray, k_pure: np.ndarray, *, kij: float = 0.72) -> float:
    n = Y.size
    k_mix = 0.0
    for i in range(n):
        correction = 0.0
        for j in range(i + 1, n):
            correction += kij * Y[j] * abs(k_pure[i] - k_pure[j])
        k_mix += Y[i] * (k_pure[i] - correction)
    return float(k_mix)


def mass_to_mole_fractions(Y: np.ndarray, mw: np.ndarray) -> np.ndarray:
    Y_arr = _as_1d_float_array("Y", Y)
    mw_arr = _as_1d_float_array("mw", mw)
    _validate_same_size(Y_arr, mw_arr, names=("Y", "mw"))
    _validate_mass_fractions(Y_arr)
    _validate_molecular_weights(mw_arr)

    numerators = Y_arr / mw_arr
    denom = float(np.sum(numerators))
    if denom <= 0.0:
        raise MixRulesValidationError("mass-to-mole conversion denominator must be positive")
    return numerators / denom


def mole_to_mass_fractions(X: np.ndarray, mw: np.ndarray) -> np.ndarray:
    X_arr = _as_1d_float_array("X", X)
    mw_arr = _as_1d_float_array("mw", mw)
    _validate_same_size(X_arr, mw_arr, names=("X", "mw"))
    _validate_mole_fractions(X_arr)
    _validate_molecular_weights(mw_arr)

    numerators = X_arr * mw_arr
    denom = float(np.sum(numerators))
    if denom <= 0.0:
        raise MixRulesValidationError("mole-to-mass conversion denominator must be positive")
    return numerators / denom


def mixture_molecular_weight_from_mass_fractions(Y: np.ndarray, mw: np.ndarray) -> float:
    Y_arr = _as_1d_float_array("Y", Y)
    mw_arr = _as_1d_float_array("mw", mw)
    _validate_same_size(Y_arr, mw_arr, names=("Y", "mw"))
    _validate_mass_fractions(Y_arr)
    _validate_molecular_weights(mw_arr)

    denom = float(np.sum(Y_arr / mw_arr))
    if denom <= 0.0:
        raise MixRulesValidationError("mixture molecular-weight denominator must be positive")
    return 1.0 / denom


def mixture_molecular_weight_from_mole_fractions(X: np.ndarray, mw: np.ndarray) -> float:
    X_arr = _as_1d_float_array("X", X)
    mw_arr = _as_1d_float_array("mw", mw)
    _validate_same_size(X_arr, mw_arr, names=("X", "mw"))
    _validate_mole_fractions(X_arr)
    _validate_molecular_weights(mw_arr)
    return float(np.sum(X_arr * mw_arr))


def mass_weighted_sum(Y: np.ndarray, values: np.ndarray) -> float:
    Y_arr = _as_1d_float_array("Y", Y)
    values_arr = _as_1d_float_array("values", values)
    _validate_same_size(Y_arr, values_arr, names=("Y", "values"))
    _validate_mass_fractions(Y_arr)
    return float(np.sum(Y_arr * values_arr))


def mole_weighted_sum(X: np.ndarray, values: np.ndarray) -> float:
    X_arr = _as_1d_float_array("X", X)
    values_arr = _as_1d_float_array("values", values)
    _validate_same_size(X_arr, values_arr, names=("X", "values"))
    _validate_mole_fractions(X_arr)
    return float(np.sum(X_arr * values_arr))


def log_mole_weighted_average(X: np.ndarray, values: np.ndarray) -> float:
    X_arr = _as_1d_float_array("X", X)
    values_arr = _as_1d_float_array("values", values)
    _validate_same_size(X_arr, values_arr, names=("X", "values"))
    _validate_mole_fractions(X_arr)
    _validate_positive_values("values", values_arr)
    return float(np.exp(np.sum(X_arr * np.log(values_arr))))


def mixture_liquid_density(
    Y: np.ndarray,
    X: np.ndarray,
    rho_pure: np.ndarray,
    *,
    model: str,
) -> float:
    Y_arr = _as_1d_float_array("Y", Y)
    X_arr = _as_1d_float_array("X", X)
    rho_arr = _as_1d_float_array("rho_pure", rho_pure)
    _validate_same_size(Y_arr, X_arr, rho_arr, names=("Y", "X", "rho_pure"))
    _validate_mass_fractions(Y_arr)
    _validate_mole_fractions(X_arr)
    _validate_positive_values("rho_pure", rho_arr)

    if model != "merino_x_sqrt_rho":
        raise MixRulesModelError(f"Unsupported liquid density mixture model: {model}")
    return float(np.square(np.sum(X_arr * np.sqrt(rho_arr))))


def mixture_liquid_cp(
    Y: np.ndarray,
    cp_pure: np.ndarray,
    *,
    model: str,
) -> float:
    Y_arr = _as_1d_float_array("Y", Y)
    cp_arr = _as_1d_float_array("cp_pure", cp_pure)
    _validate_same_size(Y_arr, cp_arr, names=("Y", "cp_pure"))
    _validate_mass_fractions(Y_arr)

    if model != "mass_weighted":
        raise MixRulesModelError(f"Unsupported liquid cp mixture model: {model}")
    return mass_weighted_sum(Y_arr, cp_arr)


def mixture_liquid_enthalpy(
    Y: np.ndarray,
    h_pure: np.ndarray,
    *,
    model: str,
) -> float:
    Y_arr = _as_1d_float_array("Y", Y)
    h_arr = _as_1d_float_array("h_pure", h_pure)
    _validate_same_size(Y_arr, h_arr, names=("Y", "h_pure"))
    _validate_mass_fractions(Y_arr)

    if model != "mass_weighted":
        raise MixRulesModelError(f"Unsupported liquid enthalpy mixture model: {model}")
    return mass_weighted_sum(Y_arr, h_arr)


def mixture_liquid_conductivity(
    Y: np.ndarray,
    X: np.ndarray,
    k_pure: np.ndarray,
    *,
    model: str,
) -> float:
    Y_arr = _as_1d_float_array("Y", Y)
    X_arr = _as_1d_float_array("X", X)
    k_arr = _as_1d_float_array("k_pure", k_pure)
    _validate_same_size(Y_arr, X_arr, k_arr, names=("Y", "X", "k_pure"))
    _validate_mass_fractions(Y_arr)
    _validate_mole_fractions(X_arr)
    _validate_positive_values("k_pure", k_arr)

    if model != "filippov":
        raise MixRulesModelError(f"Unsupported liquid conductivity mixture model: {model}")
    k_mix = _filippov_conductivity(Y_arr, k_arr)
    if not np.isfinite(k_mix) or k_mix <= 0.0:
        raise MixRulesValidationError("mixture liquid conductivity must be finite and strictly positive")
    return k_mix


def mixture_liquid_viscosity(
    Y: np.ndarray,
    X: np.ndarray,
    mu_pure: np.ndarray,
    *,
    model: str,
) -> float:
    Y_arr = _as_1d_float_array("Y", Y)
    X_arr = _as_1d_float_array("X", X)
    mu_arr = _as_1d_float_array("mu_pure", mu_pure)
    _validate_same_size(Y_arr, X_arr, mu_arr, names=("Y", "X", "mu_pure"))
    _validate_mass_fractions(Y_arr)
    _validate_mole_fractions(X_arr)
    _validate_positive_values("mu_pure", mu_arr)

    if model != "grunberg_nissan":
        raise MixRulesModelError(f"Unsupported liquid viscosity mixture model: {model}")
    return log_mole_weighted_average(X_arr, mu_arr)


def mixture_liquid_diffusivity(
    Y: np.ndarray,
    X: np.ndarray,
    *,
    model: str,
    T: float | None = None,
    mu_mix: float | None = None,
    molecular_weights: np.ndarray | None = None,
    association_factors: np.ndarray | None = None,
    molar_volumes: np.ndarray | None = None,
) -> np.ndarray | float:
    Y_arr = _as_1d_float_array("Y", Y)
    X_arr = _as_1d_float_array("X", X)
    _validate_same_size(Y_arr, X_arr, names=("Y", "X"))
    _validate_mass_fractions(Y_arr)
    _validate_mole_fractions(X_arr)

    if model != "wilke_chang":
        raise MixRulesModelError(f"Unsupported liquid diffusivity mixture model: {model}")

    if T is None:
        raise MixRulesValidationError("T must be provided for wilke_chang diffusivity")
    if mu_mix is None:
        raise MixRulesValidationError("mu_mix must be provided for wilke_chang diffusivity")
    if molecular_weights is None:
        raise MixRulesValidationError("molecular_weights must be provided for wilke_chang diffusivity")
    if association_factors is None:
        raise MixRulesValidationError("association_factors must be provided for wilke_chang diffusivity")
    if molar_volumes is None:
        raise MixRulesValidationError("molar_volumes must be provided for wilke_chang diffusivity")

    if isinstance(T, bool) or not isinstance(T, (int, float)) or not np.isfinite(T) or T <= 0.0:
        raise MixRulesValidationError("T must be a finite positive scalar")
    if (
        isinstance(mu_mix, bool)
        or not isinstance(mu_mix, (int, float))
        or not np.isfinite(mu_mix)
        or mu_mix <= 0.0
    ):
        raise MixRulesValidationError("mu_mix must be a finite positive scalar")

    mw_arr = _as_1d_float_array("molecular_weights", molecular_weights)
    phi_arr = _as_1d_float_array("association_factors", association_factors)
    volume_arr = _as_1d_float_array("molar_volumes", molar_volumes)
    _validate_same_size(
        X_arr,
        mw_arr,
        phi_arr,
        volume_arr,
        names=("X", "molecular_weights", "association_factors", "molar_volumes"),
    )
    _validate_molecular_weights(mw_arr)
    _validate_positive_values("association_factors", phi_arr)
    _validate_positive_values("molar_volumes", volume_arr)

    # paper_v1 stores molecular weight in kg/mol and molar volume in cm^3/mol.
    # The current Wilke-Chang implementation follows the engineering-unit form
    # adopted for the project:
    #   - molecular weight enters as g/mol
    #   - the summation term enters under the square root alone
    #   - temperature multiplies outside the square root
    #   - molar volume enters through (V/1000)^0.6
    #   - viscosity enters as Pa*s
    diffusivity = np.empty_like(X_arr)
    T_value = float(T)
    mu_value = float(mu_mix)
    mw_g_per_mol = mw_arr * 1000.0
    for i in range(X_arr.size):
        summation_term = float(
            np.sum(
                X_arr[np.arange(X_arr.size) != i]
                * phi_arr[np.arange(X_arr.size) != i]
                * mw_g_per_mol[np.arange(X_arr.size) != i]
            )
        )
        if summation_term < 0.0:
            raise MixRulesValidationError("Wilke-Chang summation term must be non-negative")
        diffusivity[i] = (
            1.173e-16
            * np.sqrt(summation_term)
            * T_value
            / (mu_value * ((volume_arr[i] / 1000.0) ** 0.6))
        )
    if not np.all(np.isfinite(diffusivity)) or np.any(diffusivity < 0.0):
        raise MixRulesValidationError("wilke_chang diffusivity must be finite and non-negative")
    return diffusivity


__all__ = [
    "MixRulesError",
    "MixRulesValidationError",
    "MixRulesModelError",
    "mass_to_mole_fractions",
    "mole_to_mass_fractions",
    "mixture_molecular_weight_from_mass_fractions",
    "mixture_molecular_weight_from_mole_fractions",
    "mass_weighted_sum",
    "mole_weighted_sum",
    "log_mole_weighted_average",
    "mixture_liquid_density",
    "mixture_liquid_cp",
    "mixture_liquid_enthalpy",
    "mixture_liquid_conductivity",
    "mixture_liquid_viscosity",
    "mixture_liquid_diffusivity",
]
