from __future__ import annotations

import numpy as np

from .layout import UnknownLayout
from .types import InterfaceState, SpeciesMaps, State


class StatePackError(ValueError):
    """Raised when state/vector data cannot be packed or unpacked consistently."""


def _validate_state_against_layout(
    state: State,
    layout: UnknownLayout,
    species_maps: SpeciesMaps,
) -> None:
    if state.n_liq_cells != layout.n_liq_cells:
        raise StatePackError("state.n_liq_cells must match layout.n_liq_cells")
    if state.n_gas_cells != layout.n_gas_cells:
        raise StatePackError("state.n_gas_cells must match layout.n_gas_cells")
    if state.n_liq_species_full != species_maps.n_liq_full:
        raise StatePackError("state liquid full-species count must match species_maps.n_liq_full")
    if state.n_gas_species_full != species_maps.n_gas_full:
        raise StatePackError("state gas full-species count must match species_maps.n_gas_full")
    if layout.unknowns_profile == "U_A" and species_maps.n_liq_red != 0:
        raise StatePackError("U_A layout requires species_maps.n_liq_red == 0")
    if layout.unknowns_profile == "U_B" and species_maps.n_liq_red <= 0:
        raise StatePackError("U_B layout requires species_maps.n_liq_red > 0")
    if state.interface.Ys_l_full.shape[0] != species_maps.n_liq_full:
        raise StatePackError("interface liquid full-species count must match species_maps.n_liq_full")
    if state.interface.Ys_g_full.shape[0] != species_maps.n_gas_full:
        raise StatePackError("interface gas full-species count must match species_maps.n_gas_full")


def _validate_vector_against_layout(
    vec: np.ndarray,
    layout: UnknownLayout,
) -> np.ndarray:
    arr = np.asarray(vec, dtype=np.float64)
    if arr.ndim != 1:
        raise StatePackError("state vector must be one-dimensional")
    if arr.shape[0] != layout.total_size:
        raise StatePackError("state vector length must match layout.total_size")
    return arr


def _extract_reduced_from_full(
    full_values: np.ndarray,
    full_to_reduced: np.ndarray,
    n_reduced: int,
) -> np.ndarray:
    full = np.asarray(full_values, dtype=np.float64)
    mapping = np.asarray(full_to_reduced, dtype=np.int64)
    if full.ndim != 1:
        raise StatePackError("full_values must be one-dimensional")
    if mapping.ndim != 1:
        raise StatePackError("full_to_reduced must be one-dimensional")
    if full.shape[0] != mapping.shape[0]:
        raise StatePackError("full_values length must match full_to_reduced length")

    reduced = np.zeros(n_reduced, dtype=np.float64)
    for full_idx, red_idx in enumerate(mapping):
        if red_idx >= 0:
            reduced[red_idx] = full[full_idx]
    return reduced


def _closure_full_index(
    full_names: tuple[str, ...],
    closure_name: str | None,
) -> int | None:
    if closure_name is None:
        return None
    try:
        return full_names.index(closure_name)
    except ValueError as exc:
        raise StatePackError("closure_name must belong to full_names") from exc


def _reconstruct_full_from_reduced(
    reduced_values: np.ndarray,
    reduced_to_full: np.ndarray,
    n_full: int,
    closure_full_index: int | None,
) -> np.ndarray:
    reduced = np.asarray(reduced_values, dtype=np.float64)
    mapping = np.asarray(reduced_to_full, dtype=np.int64)
    if reduced.ndim != 1:
        raise StatePackError("reduced_values must be one-dimensional")
    if mapping.ndim != 1:
        raise StatePackError("reduced_to_full must be one-dimensional")
    if reduced.shape[0] != mapping.shape[0]:
        raise StatePackError("reduced_values length must match reduced_to_full length")

    if closure_full_index is None:
        if n_full != 1 or reduced.shape[0] != 0:
            raise StatePackError("closure_full_index may be None only for single-component zero-reduced cases")
        return np.array([1.0], dtype=np.float64)

    full = np.zeros(n_full, dtype=np.float64)
    for red_idx, full_idx in enumerate(mapping):
        full[full_idx] = reduced[red_idx]
    full[closure_full_index] = 1.0 - np.sum(reduced)
    return full


def pack_state_to_array(
    state: State,
    layout: UnknownLayout,
    species_maps: SpeciesMaps,
) -> np.ndarray:
    _validate_state_against_layout(state, layout, species_maps)
    vec = np.zeros(layout.total_size, dtype=np.float64)

    for i in range(layout.n_liq_cells):
        vec[layout.liq_temperature_index(i)] = state.Tl[i]
        cell_slice = layout.liq_species_slice_for_cell(i)
        vec[cell_slice] = _extract_reduced_from_full(
            state.Yl_full[i, :],
            species_maps.liq_full_to_reduced,
            species_maps.n_liq_red,
        )

    vec[layout.if_temperature_index] = state.interface.Ts
    vec[layout.if_gas_species_slice] = _extract_reduced_from_full(
        state.interface.Ys_g_full,
        species_maps.gas_full_to_reduced,
        species_maps.n_gas_red,
    )
    vec[layout.if_mpp_index] = state.interface.mpp
    vec[layout.if_liq_species_slice] = _extract_reduced_from_full(
        state.interface.Ys_l_full,
        species_maps.liq_full_to_reduced,
        species_maps.n_liq_red,
    )

    for j in range(layout.n_gas_cells):
        vec[layout.gas_temperature_index(j)] = state.Tg[j]
        cell_slice = layout.gas_species_slice_for_cell(j)
        vec[cell_slice] = _extract_reduced_from_full(
            state.Yg_full[j, :],
            species_maps.gas_full_to_reduced,
            species_maps.n_gas_red,
        )

    return vec


def unpack_array_to_state(
    vec: np.ndarray,
    layout: UnknownLayout,
    species_maps: SpeciesMaps,
    *,
    time: float | None = None,
    state_id: str | None = None,
) -> State:
    arr = _validate_vector_against_layout(vec, layout)

    Tl = np.zeros(layout.n_liq_cells, dtype=np.float64)
    Yl_full = np.zeros((layout.n_liq_cells, species_maps.n_liq_full), dtype=np.float64)
    Tg = np.zeros(layout.n_gas_cells, dtype=np.float64)
    Yg_full = np.zeros((layout.n_gas_cells, species_maps.n_gas_full), dtype=np.float64)

    liq_closure_full_index = _closure_full_index(species_maps.liq_full_names, species_maps.liq_closure_name)
    gas_closure_full_index = _closure_full_index(species_maps.gas_full_names, species_maps.gas_closure_name)

    for i in range(layout.n_liq_cells):
        Tl[i] = arr[layout.liq_temperature_index(i)]
        Yl_full[i, :] = _reconstruct_full_from_reduced(
            arr[layout.liq_species_slice_for_cell(i)],
            species_maps.liq_reduced_to_full,
            species_maps.n_liq_full,
            liq_closure_full_index,
        )

    Ys_g_full = _reconstruct_full_from_reduced(
        arr[layout.if_gas_species_slice],
        species_maps.gas_reduced_to_full,
        species_maps.n_gas_full,
        gas_closure_full_index,
    )
    Ys_l_full = _reconstruct_full_from_reduced(
        arr[layout.if_liq_species_slice],
        species_maps.liq_reduced_to_full,
        species_maps.n_liq_full,
        liq_closure_full_index,
    )
    interface = InterfaceState(
        Ts=float(arr[layout.if_temperature_index]),
        mpp=float(arr[layout.if_mpp_index]),
        Ys_g_full=Ys_g_full,
        Ys_l_full=Ys_l_full,
    )

    for j in range(layout.n_gas_cells):
        Tg[j] = arr[layout.gas_temperature_index(j)]
        Yg_full[j, :] = _reconstruct_full_from_reduced(
            arr[layout.gas_species_slice_for_cell(j)],
            species_maps.gas_reduced_to_full,
            species_maps.n_gas_full,
            gas_closure_full_index,
        )

    return State(
        Tl=Tl,
        Yl_full=Yl_full,
        Tg=Tg,
        Yg_full=Yg_full,
        interface=interface,
        rho_l=None,
        rho_g=None,
        hl=None,
        hg=None,
        Xg_full=None,
        time=time,
        state_id=state_id,
    )


def apply_trial_vector_to_state(
    base_state: State,
    vec_trial: np.ndarray,
    layout: UnknownLayout,
    species_maps: SpeciesMaps,
) -> State:
    trial = unpack_array_to_state(
        vec_trial,
        layout,
        species_maps,
        time=base_state.time,
        state_id=base_state.state_id,
    )
    return trial


def extract_block_views(
    vec: np.ndarray,
    layout: UnknownLayout,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    arr = _validate_vector_against_layout(vec, layout)
    return arr[layout.liq_block], arr[layout.if_block], arr[layout.gas_block]


def reshape_bulk_block_views(
    vec: np.ndarray,
    layout: UnknownLayout,
) -> tuple[np.ndarray, np.ndarray]:
    liq_block, _, gas_block = extract_block_views(vec, layout)
    liq_cells = liq_block.reshape(layout.n_liq_cells, layout.liq_cell_width)
    gas_cells = gas_block.reshape(layout.n_gas_cells, layout.gas_cell_width)
    return liq_cells, gas_cells


__all__ = [
    "StatePackError",
    "apply_trial_vector_to_state",
    "extract_block_views",
    "pack_state_to_array",
    "reshape_bulk_block_views",
    "unpack_array_to_state",
]
