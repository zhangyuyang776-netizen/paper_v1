from __future__ import annotations

"""Distributed-safe export and validation of layout-driven PETSc fieldsplit IS."""

from dataclasses import dataclass


class FieldSplitExportError(RuntimeError):
    """Raised when layout-driven fieldsplit export or validation fails."""


@dataclass(slots=True)
class FieldSplitISRecord:
    """One exported layout-driven fieldsplit with owned-count diagnostics."""

    name: str
    blocks: tuple[str, ...]
    policy: str
    is_obj: object
    global_index_count: int
    owned_index_count: int


def _require_layout_fieldsplit_api(layout: object) -> None:
    for attr in ("default_fieldsplit_plan", "describe_fieldsplits", "build_is_petsc"):
        if not hasattr(layout, attr):
            raise FieldSplitExportError(f"layout missing required fieldsplit API: {attr}")


def _normalize_plan(plan: str | None) -> str:
    plan_name = "bulk_iface" if plan is None else str(plan).strip()
    if plan_name != "bulk_iface":
        raise FieldSplitExportError(f"unsupported fieldsplit plan: {plan_name}")
    return plan_name


def _owned_count_from_is(is_obj: object) -> int:
    get_local_size = getattr(is_obj, "getLocalSize", None)
    if callable(get_local_size):
        return int(get_local_size())
    get_indices = getattr(is_obj, "getIndices", None)
    if callable(get_indices):
        return len(tuple(get_indices()))
    indices = getattr(is_obj, "indices", None)
    if indices is not None:
        return len(tuple(indices))
    raise FieldSplitExportError("PETSc IS object does not expose local size information")


def export_fieldsplit_is(
    *,
    layout: object,
    PETSc: object,
    plan: str = "bulk_iface",
    ownership_range: tuple[int, int] | None = None,
) -> dict[str, object]:
    """Export layout-driven PETSc IS objects for the requested fieldsplit plan."""

    _require_layout_fieldsplit_api(layout)
    plan_name = _normalize_plan(plan)
    try:
        return dict(layout.build_is_petsc(PETSc=PETSc, plan=plan_name, ownership_range=ownership_range))
    except Exception as exc:
        raise FieldSplitExportError(f"layout.build_is_petsc(...) failed: {exc}") from exc


def validate_fieldsplit_is_records(
    records: tuple[FieldSplitISRecord, ...],
    *,
    layout: object,
    plan: str = "bulk_iface",
    ownership_range: tuple[int, int] | None = None,
) -> None:
    """Validate exported split names, counts, and iface/Rd invariants."""

    _ = ownership_range
    _require_layout_fieldsplit_api(layout)
    plan_name = _normalize_plan(plan)

    expected = tuple(layout.default_fieldsplit_plan(plan_name))
    expected_names = tuple(str(item["name"]) for item in expected)
    got_names = tuple(rec.name for rec in records)
    if got_names != expected_names:
        raise FieldSplitExportError(f"split name/order mismatch: expected {expected_names}, got {got_names}")

    desc_map = {str(item["name"]): dict(item) for item in tuple(layout.describe_fieldsplits(plan_name))}
    seen: set[str] = set()
    for rec in records:
        if rec.name in seen:
            raise FieldSplitExportError(f"duplicate split name: {rec.name}")
        seen.add(rec.name)
        if rec.name not in desc_map:
            raise FieldSplitExportError(f"missing layout description for split '{rec.name}'")
        if rec.name == "iface" and any(block.strip().lower() == "rd" for block in rec.blocks):
            raise FieldSplitExportError("iface split must not include Rd")

        expected_count = int(desc_map[rec.name]["index_count"])
        if rec.global_index_count != expected_count:
            raise FieldSplitExportError(
                f"global index count mismatch for split '{rec.name}': {rec.global_index_count} != {expected_count}"
            )
        if rec.owned_index_count < 0 or rec.owned_index_count > rec.global_index_count:
            raise FieldSplitExportError(
                f"invalid owned count for split '{rec.name}': {rec.owned_index_count}"
            )


def build_fieldsplit_is_records(
    *,
    layout: object,
    PETSc: object,
    plan: str = "bulk_iface",
    ownership_range: tuple[int, int] | None = None,
) -> tuple[FieldSplitISRecord, ...]:
    """Build structured fieldsplit records from layout metadata and exported PETSc IS."""

    _require_layout_fieldsplit_api(layout)
    plan_name = _normalize_plan(plan)
    raw_plan = tuple(layout.default_fieldsplit_plan(plan_name))
    descriptions = {str(item["name"]): dict(item) for item in tuple(layout.describe_fieldsplits(plan_name))}
    is_map = export_fieldsplit_is(
        layout=layout,
        PETSc=PETSc,
        plan=plan_name,
        ownership_range=ownership_range,
    )

    records: list[FieldSplitISRecord] = []
    for item in raw_plan:
        name = str(item["name"])
        if name not in is_map:
            raise FieldSplitExportError(f"missing IS for split '{name}'")
        desc = descriptions.get(name)
        if desc is None:
            raise FieldSplitExportError(f"missing description for split '{name}'")
        records.append(
            FieldSplitISRecord(
                name=name,
                blocks=tuple(str(block) for block in tuple(item["blocks"])),
                policy=str(item["policy"]),
                is_obj=is_map[name],
                global_index_count=int(desc["index_count"]),
                owned_index_count=_owned_count_from_is(is_map[name]),
            )
        )

    records_tuple = tuple(records)
    validate_fieldsplit_is_records(
        records_tuple,
        layout=layout,
        plan=plan_name,
        ownership_range=ownership_range,
    )
    return records_tuple


def describe_owned_fieldsplits(
    *,
    layout: object,
    PETSc: object,
    plan: str = "bulk_iface",
    ownership_range: tuple[int, int] | None = None,
) -> tuple[dict[str, object], ...]:
    """Return per-split global and owned index counts for diagnostics."""

    records = build_fieldsplit_is_records(
        layout=layout,
        PETSc=PETSc,
        plan=plan,
        ownership_range=ownership_range,
    )
    return tuple(
        {
            "name": rec.name,
            "blocks": rec.blocks,
            "policy": rec.policy,
            "global_index_count": rec.global_index_count,
            "owned_index_count": rec.owned_index_count,
        }
        for rec in records
    )


def build_fieldsplit_handles(
    *,
    layout: object,
    PETSc: object,
    ownership_range: tuple[int, int] | None = None,
    plan: str = "bulk_iface",
) -> dict[str, object]:
    """Build a compact fieldsplit export bundle for diagnostics or future hooks."""

    plan_name = _normalize_plan(plan)
    return {
        "fieldsplit_plan": plan_name,
        "fieldsplit_is_map": export_fieldsplit_is(
            layout=layout,
            PETSc=PETSc,
            plan=plan_name,
            ownership_range=ownership_range,
        ),
        "fieldsplit_records": build_fieldsplit_is_records(
            layout=layout,
            PETSc=PETSc,
            plan=plan_name,
            ownership_range=ownership_range,
        ),
    }


def build_fieldsplit_specs_from_parallel(
    *,
    layout: object,
    PETSc: object,
    ownership_range: tuple[int, int] | None = None,
    plan: str = "bulk_iface",
) -> tuple[dict[str, object], ...]:
    """Return spec-like dictionaries without importing solver-internal dataclasses."""

    return tuple(
        {
            "name": rec.name,
            "blocks": rec.blocks,
            "policy": rec.policy,
            "is_obj": rec.is_obj,
            "index_count": rec.global_index_count,
        }
        for rec in build_fieldsplit_is_records(
            layout=layout,
            PETSc=PETSc,
            plan=plan,
            ownership_range=ownership_range,
        )
    )


__all__ = [
    "FieldSplitExportError",
    "FieldSplitISRecord",
    "build_fieldsplit_handles",
    "build_fieldsplit_is_records",
    "build_fieldsplit_specs_from_parallel",
    "describe_owned_fieldsplits",
    "export_fieldsplit_is",
    "validate_fieldsplit_is_records",
]
