from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .types import Mesh1D, RunConfig, SpeciesMaps


@dataclass(slots=True, kw_only=True, frozen=True)
class UnknownLayout:
    """Frozen inner-unknown layout for one fixed-geometry nonlinear solve.

    Bulk temperature slices are strided global slices. Bulk species slices are
    local-in-cell templates; use the per-cell helper methods for exact global
    species ranges.
    """

    unknowns_profile: str

    n_liq_cells: int
    n_gas_cells: int

    n_liq_red: int
    n_gas_red: int

    liq_cell_width: int
    gas_cell_width: int
    n_if_unknowns: int

    liq_block: slice
    if_block: slice
    gas_block: slice

    total_size: int

    liq_temperature_slice: slice
    liq_species_local_slice: slice

    if_temperature_slice: slice
    if_gas_species_slice: slice
    if_mpp_slice: slice
    if_liq_species_slice: slice

    gas_temperature_slice: slice
    gas_species_local_slice: slice

    def __post_init__(self) -> None:
        if self.unknowns_profile not in {"U_A", "U_B"}:
            raise ValueError("unknowns_profile must be one of {'U_A', 'U_B'}")
        if self.n_liq_cells <= 0:
            raise ValueError("n_liq_cells must be > 0")
        if self.n_gas_cells <= 0:
            raise ValueError("n_gas_cells must be > 0")
        if self.liq_cell_width <= 0:
            raise ValueError("liq_cell_width must be > 0")
        if self.gas_cell_width <= 0:
            raise ValueError("gas_cell_width must be > 0")
        if self.n_if_unknowns <= 0:
            raise ValueError("n_if_unknowns must be > 0")

        if self.liq_block.start != 0:
            raise ValueError("liq_block must start at 0")
        if self.liq_block.stop != self.if_block.start:
            raise ValueError("liq_block and if_block must be contiguous")
        if self.if_block.stop != self.gas_block.start:
            raise ValueError("if_block and gas_block must be contiguous")
        if self.gas_block.stop != self.total_size:
            raise ValueError("gas_block must end at total_size")
        if self.liq_block.stop - self.liq_block.start != self.n_liq_cells * self.liq_cell_width:
            raise ValueError("liq_block length must equal n_liq_cells * liq_cell_width")
        if self.if_block.stop - self.if_block.start != self.n_if_unknowns:
            raise ValueError("if_block length must equal n_if_unknowns")
        if self.gas_block.stop - self.gas_block.start != self.n_gas_cells * self.gas_cell_width:
            raise ValueError("gas_block length must equal n_gas_cells * gas_cell_width")

        if self.if_temperature_slice.stop - self.if_temperature_slice.start != 1:
            raise ValueError("if_temperature_slice must have length 1")
        if self.if_mpp_slice.stop - self.if_mpp_slice.start != 1:
            raise ValueError("if_mpp_slice must have length 1")
        if self.if_gas_species_slice.stop - self.if_gas_species_slice.start != self.n_gas_red:
            raise ValueError("if_gas_species_slice length must match n_gas_red")
        if self.if_liq_species_slice.stop - self.if_liq_species_slice.start != self.n_liq_red:
            raise ValueError("if_liq_species_slice length must match n_liq_red")
        if self.if_temperature_slice.start != self.if_block.start:
            raise ValueError("if_temperature_slice must start at if_block.start")
        if self.if_mpp_slice.start != self.if_gas_species_slice.stop:
            raise ValueError("if_mpp_slice must follow if_gas_species_slice")
        if self.if_liq_species_slice.start != self.if_mpp_slice.stop:
            raise ValueError("if_liq_species_slice must follow if_mpp_slice")
        if self.if_liq_species_slice.stop != self.if_block.stop:
            raise ValueError("interface field slices must fill if_block exactly")

        if self.liq_temperature_slice.start != self.liq_block.start:
            raise ValueError("liq_temperature_slice must start at liq_block.start")
        if self.liq_temperature_slice.stop != self.liq_block.stop:
            raise ValueError("liq_temperature_slice must stop at liq_block.stop")
        if self.liq_temperature_slice.step != self.liq_cell_width:
            raise ValueError("liq_temperature_slice step must equal liq_cell_width")
        if self.gas_temperature_slice.start != self.gas_block.start:
            raise ValueError("gas_temperature_slice must start at gas_block.start")
        if self.gas_temperature_slice.stop != self.gas_block.stop:
            raise ValueError("gas_temperature_slice must stop at gas_block.stop")
        if self.gas_temperature_slice.step != self.gas_cell_width:
            raise ValueError("gas_temperature_slice step must equal gas_cell_width")

        if self.unknowns_profile == "U_A" and self.n_liq_red != 0:
            raise ValueError("U_A layout must have n_liq_red == 0")
        if self.unknowns_profile == "U_B" and self.n_liq_red <= 0:
            raise ValueError("U_B layout must have n_liq_red > 0")

    @property
    def has_liq_species_bulk(self) -> bool:
        return self.n_liq_red > 0

    @property
    def has_liq_species_interface(self) -> bool:
        return self.n_liq_red > 0

    @property
    def has_gas_species(self) -> bool:
        return self.n_gas_red > 0

    @property
    def n_total_unknowns(self) -> int:
        return self.total_size

    @property
    def if_temperature_index(self) -> int:
        return self.if_temperature_slice.start

    @property
    def if_mpp_index(self) -> int:
        return self.if_mpp_slice.start

    def liq_cell_slice(self, i: int) -> slice:
        if not (0 <= i < self.n_liq_cells):
            raise IndexError("liquid cell index out of range")
        start = self.liq_block.start + i * self.liq_cell_width
        return slice(start, start + self.liq_cell_width)

    def liq_temperature_index(self, i: int) -> int:
        return self.liq_cell_slice(i).start

    def liq_species_slice_for_cell(self, i: int) -> slice:
        cell = self.liq_cell_slice(i)
        start = cell.start + 1
        return slice(start, start + self.n_liq_red)

    def gas_cell_slice(self, j: int) -> slice:
        if not (0 <= j < self.n_gas_cells):
            raise IndexError("gas cell index out of range")
        start = self.gas_block.start + j * self.gas_cell_width
        return slice(start, start + self.gas_cell_width)

    def gas_temperature_index(self, j: int) -> int:
        return self.gas_cell_slice(j).start

    def gas_species_slice_for_cell(self, j: int) -> slice:
        cell = self.gas_cell_slice(j)
        start = cell.start + 1
        return slice(start, start + self.n_gas_red)

    def has_block(self, block_name: str) -> bool:
        """Return whether the frozen inner layout exposes a named block."""

        key = block_name.strip().lower()
        if key in {"liq", "liquid"}:
            return True
        if key in {"if", "iface", "interface"}:
            return True
        if key == "gas":
            return True
        if key in {"ts", "if_temperature"}:
            return True
        if key in {"mpp", "if_mpp"}:
            return True
        if key in {"if_gas_species", "gas_species_interface"}:
            return self.n_gas_red > 0
        if key in {"if_liq_species", "liq_species_interface"}:
            return self.n_liq_red > 0
        if key == "rd":
            return False
        return False

    def _global_indices_for_block(self, block_name: str) -> tuple[int, ...]:
        key = block_name.strip().lower()
        if key in {"liq", "liquid"}:
            return tuple(range(self.liq_block.start, self.liq_block.stop))
        if key in {"if", "iface", "interface"}:
            return tuple(range(self.if_block.start, self.if_block.stop))
        if key == "gas":
            return tuple(range(self.gas_block.start, self.gas_block.stop))
        raise ValueError(f"unsupported layout block for fieldsplit export: {block_name}")

    def default_fieldsplit_plan(self, plan: str = "bulk_iface") -> tuple[dict[str, object], ...]:
        """Return the default layout-driven fieldsplit plan."""

        if plan != "bulk_iface":
            raise ValueError(f"unsupported fieldsplit plan: {plan}")
        return (
            {"name": "bulk", "blocks": ("liq", "gas"), "policy": "concat"},
            {"name": "iface", "blocks": ("if",), "policy": "contiguous"},
        )

    def describe_fieldsplits(self, plan: str = "bulk_iface") -> tuple[dict[str, object], ...]:
        """Describe split membership and global index counts for diagnostics."""

        descriptions: list[dict[str, object]] = []
        for spec in self.default_fieldsplit_plan(plan):
            blocks = tuple(str(name) for name in spec["blocks"])
            indices: list[int] = []
            for block_name in blocks:
                indices.extend(self._global_indices_for_block(block_name))
            descriptions.append(
                {
                    "name": str(spec["name"]),
                    "blocks": blocks,
                    "policy": str(spec["policy"]),
                    "index_count": len(indices),
                }
            )
        return tuple(descriptions)

    def build_is_petsc(
        self,
        *,
        PETSc: Any,
        plan: str = "bulk_iface",
        ownership_range: tuple[int, int] | None = None,
    ) -> dict[str, object]:
        """Build PETSc IS objects for the requested fieldsplit plan."""

        is_map: dict[str, object] = {}
        rstart, rend = (None, None) if ownership_range is None else ownership_range
        for spec in self.default_fieldsplit_plan(plan):
            indices: list[int] = []
            for block_name in tuple(spec["blocks"]):
                indices.extend(self._global_indices_for_block(str(block_name)))
            if rstart is not None and rend is not None:
                indices = [idx for idx in indices if int(rstart) <= idx < int(rend)]
            is_map[str(spec["name"])] = PETSc.IS().createGeneral(indices)
        return is_map


def _validate_layout_inputs(run_cfg: RunConfig, mesh: Mesh1D) -> None:
    if run_cfg.unknowns_profile not in {"U_A", "U_B"}:
        raise ValueError("run_cfg.unknowns_profile must be one of {'U_A', 'U_B'}")
    if mesh.n_liq <= 0:
        raise ValueError("mesh.n_liq must be > 0")
    if mesh.n_gas <= 0:
        raise ValueError("mesh.n_gas must be > 0")

    species_maps = run_cfg.species_maps
    if run_cfg.unknowns_profile == "U_A" and species_maps.n_liq_red != 0:
        raise ValueError("U_A layout requires species_maps.n_liq_red == 0")
    if run_cfg.unknowns_profile == "U_B" and species_maps.n_liq_red <= 0:
        raise ValueError("U_B layout requires species_maps.n_liq_red > 0")


def _derive_liq_cell_width(species_maps: SpeciesMaps, unknowns_profile: str) -> int:
    if unknowns_profile == "U_A":
        return 1
    return 1 + species_maps.n_liq_red


def _derive_gas_cell_width(species_maps: SpeciesMaps) -> int:
    return 1 + species_maps.n_gas_red


def _derive_interface_width(species_maps: SpeciesMaps, unknowns_profile: str) -> int:
    if unknowns_profile == "U_A":
        return 1 + species_maps.n_gas_red + 1
    return 1 + species_maps.n_gas_red + 1 + species_maps.n_liq_red


def _build_block_slices(
    *,
    n_liq_cells: int,
    n_gas_cells: int,
    liq_cell_width: int,
    gas_cell_width: int,
    n_if_unknowns: int,
) -> tuple[slice, slice, slice, int]:
    liq_size = n_liq_cells * liq_cell_width
    gas_size = n_gas_cells * gas_cell_width
    liq_block = slice(0, liq_size)
    if_block = slice(liq_block.stop, liq_block.stop + n_if_unknowns)
    gas_block = slice(if_block.stop, if_block.stop + gas_size)
    total_size = gas_block.stop
    return liq_block, if_block, gas_block, total_size


def _build_field_slices(
    *,
    liq_block: slice,
    if_block: slice,
    gas_block: slice,
    liq_cell_width: int,
    gas_cell_width: int,
    n_liq_red: int,
    n_gas_red: int,
) -> dict[str, slice]:
    if_start = if_block.start
    if_temperature_slice = slice(if_start, if_start + 1)
    if_gas_species_slice = slice(if_temperature_slice.stop, if_temperature_slice.stop + n_gas_red)
    if_mpp_slice = slice(if_gas_species_slice.stop, if_gas_species_slice.stop + 1)
    if_liq_species_slice = slice(if_mpp_slice.stop, if_mpp_slice.stop + n_liq_red)

    return {
        "liq_temperature_slice": slice(liq_block.start, liq_block.stop, liq_cell_width),
        "liq_species_local_slice": slice(1, 1 + n_liq_red),
        "if_temperature_slice": if_temperature_slice,
        "if_gas_species_slice": if_gas_species_slice,
        "if_mpp_slice": if_mpp_slice,
        "if_liq_species_slice": if_liq_species_slice,
        "gas_temperature_slice": slice(gas_block.start, gas_block.stop, gas_cell_width),
        "gas_species_local_slice": slice(1, 1 + n_gas_red),
    }


def build_layout(run_cfg: RunConfig, mesh: Mesh1D) -> UnknownLayout:
    _validate_layout_inputs(run_cfg, mesh)
    species_maps = run_cfg.species_maps

    n_liq_red = species_maps.n_liq_red
    n_gas_red = species_maps.n_gas_red
    liq_cell_width = _derive_liq_cell_width(species_maps, run_cfg.unknowns_profile)
    gas_cell_width = _derive_gas_cell_width(species_maps)
    n_if_unknowns = _derive_interface_width(species_maps, run_cfg.unknowns_profile)

    liq_block, if_block, gas_block, total_size = _build_block_slices(
        n_liq_cells=mesh.n_liq,
        n_gas_cells=mesh.n_gas,
        liq_cell_width=liq_cell_width,
        gas_cell_width=gas_cell_width,
        n_if_unknowns=n_if_unknowns,
    )
    field_slices = _build_field_slices(
        liq_block=liq_block,
        if_block=if_block,
        gas_block=gas_block,
        liq_cell_width=liq_cell_width,
        gas_cell_width=gas_cell_width,
        n_liq_red=n_liq_red,
        n_gas_red=n_gas_red,
    )

    return UnknownLayout(
        unknowns_profile=run_cfg.unknowns_profile,
        n_liq_cells=mesh.n_liq,
        n_gas_cells=mesh.n_gas,
        n_liq_red=n_liq_red,
        n_gas_red=n_gas_red,
        liq_cell_width=liq_cell_width,
        gas_cell_width=gas_cell_width,
        n_if_unknowns=n_if_unknowns,
        liq_block=liq_block,
        if_block=if_block,
        gas_block=gas_block,
        total_size=total_size,
        liq_temperature_slice=field_slices["liq_temperature_slice"],
        liq_species_local_slice=field_slices["liq_species_local_slice"],
        if_temperature_slice=field_slices["if_temperature_slice"],
        if_gas_species_slice=field_slices["if_gas_species_slice"],
        if_mpp_slice=field_slices["if_mpp_slice"],
        if_liq_species_slice=field_slices["if_liq_species_slice"],
        gas_temperature_slice=field_slices["gas_temperature_slice"],
        gas_species_local_slice=field_slices["gas_species_local_slice"],
    )


__all__ = ["UnknownLayout", "build_layout"]
