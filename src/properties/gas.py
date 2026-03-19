from __future__ import annotations

"""Gas thermo / transport backend for Phase 2."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cantera as ct
import numpy as np


class GasThermoError(Exception):
    """Base error for gas thermo backend construction and evaluation."""


class GasThermoValidationError(GasThermoError):
    """Raised when gas thermo inputs are structurally invalid."""


class GasThermoModelError(GasThermoError):
    """Raised when the Cantera gas backend or mechanism contract is inconsistent."""


class GasEnthalpyModelError(GasThermoError):
    """Raised when gas enthalpy evaluation fails or returns non-finite values."""


def _as_1d_float_array(name: str, value: Any, *, expected_size: int | None = None) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim != 1:
        raise GasThermoValidationError(f"{name} must be a one-dimensional float array")
    if arr.size == 0:
        raise GasThermoValidationError(f"{name} must be non-empty")
    if not np.all(np.isfinite(arr)):
        raise GasThermoValidationError(f"{name} must contain only finite values")
    if expected_size is not None and arr.size != expected_size:
        raise GasThermoValidationError(f"{name} must have length {expected_size}")
    return arr


def _validate_temperature(T: float) -> float:
    if isinstance(T, bool) or not np.isscalar(T):
        raise GasThermoValidationError("T must be a finite positive scalar")
    T_value = float(T)
    if not np.isfinite(T_value) or T_value <= 0.0:
        raise GasThermoValidationError("T must be a finite positive scalar")
    return T_value


def _validate_pressure(P: float) -> float:
    if isinstance(P, bool) or not np.isscalar(P):
        raise GasThermoValidationError("P must be a finite positive scalar")
    P_value = float(P)
    if not np.isfinite(P_value) or P_value <= 0.0:
        raise GasThermoValidationError("P must be a finite positive scalar")
    return P_value


def _validate_mass_fractions(Y_full: np.ndarray, n_spec: int, *, atol: float = 1.0e-12) -> np.ndarray:
    Y = _as_1d_float_array("Y_full", Y_full, expected_size=n_spec)
    if np.any(Y < 0.0):
        raise GasThermoValidationError("Y_full must be non-negative")
    if not np.isclose(float(np.sum(Y)), 1.0, rtol=0.0, atol=atol):
        raise GasThermoValidationError("Y_full must sum to 1 within tolerance")
    return Y


def _validate_batch_inputs(
    T: np.ndarray,
    Y_full_2d: np.ndarray,
    P: np.ndarray | float,
    *,
    n_spec: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    T_arr = np.asarray(T, dtype=np.float64)
    Y_arr = np.asarray(Y_full_2d, dtype=np.float64)
    if T_arr.ndim != 1:
        raise GasThermoValidationError("T batch input must be one-dimensional")
    if Y_arr.ndim != 2:
        raise GasThermoValidationError("Y_full_2d must be two-dimensional")
    if Y_arr.shape[0] != T_arr.shape[0]:
        raise GasThermoValidationError("Y_full_2d row count must match T length")
    if Y_arr.shape[1] != n_spec:
        raise GasThermoValidationError(f"Y_full_2d species dimension must be {n_spec}")
    if not np.all(np.isfinite(T_arr)):
        raise GasThermoValidationError("T batch input must contain only finite values")
    if not np.all(np.isfinite(Y_arr)):
        raise GasThermoValidationError("Y_full_2d must contain only finite values")

    if np.isscalar(P):
        P_arr = np.full(T_arr.shape[0], _validate_pressure(float(P)), dtype=np.float64)
    else:
        P_arr = np.asarray(P, dtype=np.float64)
        if P_arr.ndim != 1:
            raise GasThermoValidationError("P batch input must be a scalar or one-dimensional array")
        if P_arr.shape[0] != T_arr.shape[0]:
            raise GasThermoValidationError("P batch vector length must match T length")
        if not np.all(np.isfinite(P_arr)) or np.any(P_arr <= 0.0):
            raise GasThermoValidationError("P batch input must contain only finite positive values")
    return T_arr, Y_arr, P_arr


@dataclass(frozen=True, kw_only=True)
class GasThermoModel:
    mechanism_path: str
    species_names: tuple[str, ...]
    molecular_weights: np.ndarray
    closure_species: str
    reference_pressure: float = ct.one_atm
    transport_model: str | None = None
    _gas: ct.Solution = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if len(self.species_names) == 0:
            raise GasThermoValidationError("species_names must be non-empty")
        if len(set(self.species_names)) != len(self.species_names):
            raise GasThermoValidationError("species_names must not contain duplicates")
        if not self.closure_species:
            raise GasThermoValidationError("closure_species must be non-empty")
        if self.closure_species not in self.species_names:
            raise GasThermoValidationError("closure_species must be in species_names")

        mw = _as_1d_float_array("molecular_weights", self.molecular_weights, expected_size=len(self.species_names))
        if np.any(mw <= 0.0):
            raise GasThermoValidationError("molecular_weights must be strictly positive")
        mw.setflags(write=False)
        object.__setattr__(self, "molecular_weights", mw)

        ref_P = _validate_pressure(self.reference_pressure)
        object.__setattr__(self, "reference_pressure", ref_P)

        if tuple(self._gas.species_names) != self.species_names:
            raise GasThermoModelError("GasThermoModel species_names must exactly match the mechanism species order")
        cantera_mw = np.asarray(self._gas.molecular_weights, dtype=np.float64) / 1000.0
        if not np.allclose(cantera_mw, self.molecular_weights, rtol=1.0e-10, atol=1.0e-12):
            raise GasThermoModelError("molecular_weights must match the mechanism molecular weights in kg/mol")

        if self.transport_model is None:
            object.__setattr__(self, "transport_model", self._gas.transport_model)

    @property
    def n_species(self) -> int:
        return len(self.species_names)

    def _set_tpy(self, T: float, P: float, Y_full: np.ndarray) -> np.ndarray:
        T_value = _validate_temperature(T)
        P_value = _validate_pressure(P)
        Y = _validate_mass_fractions(Y_full, self.n_species)
        try:
            self._gas.TPY = T_value, P_value, Y
        except ct.CanteraError as exc:
            raise GasThermoModelError(f"Failed to set gas state TPY: {exc}") from exc
        return Y

    def mixture_molecular_weight(self, Y_full: np.ndarray) -> float:
        Y = _validate_mass_fractions(Y_full, self.n_species)
        denom = float(np.sum(Y / self.molecular_weights))
        if denom <= 0.0:
            raise GasThermoValidationError("mixture molecular-weight denominator must be positive")
        return 1.0 / denom

    def density_mass(self, T: float, Y_full: np.ndarray, P: float) -> float:
        self._set_tpy(T, P, Y_full)
        rho = float(self._gas.density_mass)
        if not np.isfinite(rho) or rho <= 0.0:
            raise GasThermoModelError("gas density must be finite and positive")
        return rho

    def cp_mass(self, T: float, Y_full: np.ndarray, P: float | None = None) -> float:
        self._set_tpy(T, self.reference_pressure if P is None else P, Y_full)
        cp = float(self._gas.cp_mass)
        if not np.isfinite(cp) or cp <= 0.0:
            raise GasThermoModelError("gas cp_mass must be finite and positive")
        return cp

    def enthalpy_mass(self, T: float, Y_full: np.ndarray, P: float | None = None) -> float:
        self._set_tpy(T, self.reference_pressure if P is None else P, Y_full)
        h = float(self._gas.enthalpy_mass)
        if not np.isfinite(h):
            raise GasEnthalpyModelError("gas enthalpy_mass must be finite")
        return h

    def conductivity(self, T: float, Y_full: np.ndarray, P: float) -> float:
        self._set_tpy(T, P, Y_full)
        k = float(self._gas.thermal_conductivity)
        if not np.isfinite(k) or k <= 0.0:
            raise GasThermoModelError("gas thermal conductivity must be finite and positive")
        return k

    def viscosity(self, T: float, Y_full: np.ndarray, P: float) -> float:
        self._set_tpy(T, P, Y_full)
        mu = float(self._gas.viscosity)
        if not np.isfinite(mu) or mu <= 0.0:
            raise GasThermoModelError("gas viscosity must be finite and positive")
        return mu

    def diffusivity(self, T: float, Y_full: np.ndarray, P: float) -> np.ndarray:
        self._set_tpy(T, P, Y_full)
        D = np.asarray(self._gas.mix_diff_coeffs_mass, dtype=np.float64)
        if D.ndim != 1 or D.shape[0] != self.n_species:
            raise GasThermoModelError("gas mixture diffusion vector has inconsistent shape")
        if not np.all(np.isfinite(D)) or np.any(D <= 0.0):
            raise GasThermoModelError("gas mixture diffusion vector must be finite and strictly positive")
        return D.copy()

    def species_enthalpies_mass(self, T: float) -> np.ndarray:
        T_value = _validate_temperature(T)
        try:
            self._gas.TP = T_value, self.reference_pressure
        except ct.CanteraError as exc:
            raise GasThermoModelError(f"Failed to set gas state TP for species enthalpies: {exc}") from exc
        h_species = (
            np.asarray(self._gas.standard_enthalpies_RT, dtype=np.float64)
            * ct.gas_constant
            * T_value
            / np.asarray(self._gas.molecular_weights, dtype=np.float64)
        )
        if not np.all(np.isfinite(h_species)):
            raise GasEnthalpyModelError("gas species enthalpies must be finite")
        return h_species

    def density_mass_batch(self, T: np.ndarray, Y_full_2d: np.ndarray, P: np.ndarray | float) -> np.ndarray:
        T_arr, Y_arr, P_arr = _validate_batch_inputs(T, Y_full_2d, P, n_spec=self.n_species)
        return np.array(
            [self.density_mass(float(T_i), Y_arr[i, :], float(P_arr[i])) for i, T_i in enumerate(T_arr)],
            dtype=np.float64,
        )

    def cp_mass_batch(self, T: np.ndarray, Y_full_2d: np.ndarray, P: np.ndarray | float) -> np.ndarray:
        T_arr, Y_arr, P_arr = _validate_batch_inputs(T, Y_full_2d, P, n_spec=self.n_species)
        return np.array(
            [self.cp_mass(float(T_i), Y_arr[i, :], float(P_arr[i])) for i, T_i in enumerate(T_arr)],
            dtype=np.float64,
        )

    def enthalpy_mass_batch(self, T: np.ndarray, Y_full_2d: np.ndarray, P: np.ndarray | float) -> np.ndarray:
        T_arr, Y_arr, P_arr = _validate_batch_inputs(T, Y_full_2d, P, n_spec=self.n_species)
        return np.array(
            [self.enthalpy_mass(float(T_i), Y_arr[i, :], float(P_arr[i])) for i, T_i in enumerate(T_arr)],
            dtype=np.float64,
        )


def build_gas_thermo_model(
    *,
    mechanism_path: str | Path,
    gas_species_full: tuple[str, ...],
    molecular_weights: np.ndarray,
    closure_species: str,
) -> GasThermoModel:
    mech = str(mechanism_path)
    try:
        gas = ct.Solution(mech)
    except ct.CanteraError as exc:
        raise GasThermoModelError(f"Failed to load gas mechanism {mech!r}: {exc}") from exc

    if tuple(gas.species_names) != tuple(gas_species_full):
        raise GasThermoModelError("gas_species_full must exactly match the mechanism full species order")
    if closure_species not in gas_species_full:
        raise GasThermoValidationError("closure_species must be present in gas_species_full")

    mw = _as_1d_float_array("molecular_weights", molecular_weights, expected_size=len(gas_species_full))
    cantera_mw = np.asarray(gas.molecular_weights, dtype=np.float64) / 1000.0
    if not np.allclose(mw, cantera_mw, rtol=1.0e-10, atol=1.0e-12):
        raise GasThermoModelError("molecular_weights must match the mechanism molecular weights in kg/mol")

    return GasThermoModel(
        mechanism_path=mech,
        species_names=tuple(gas_species_full),
        molecular_weights=mw.copy(),
        closure_species=closure_species,
        transport_model=gas.transport_model,
        _gas=gas,
    )


__all__ = [
    "GasEnthalpyModelError",
    "GasThermoError",
    "GasThermoModel",
    "GasThermoModelError",
    "GasThermoValidationError",
    "build_gas_thermo_model",
]
