from __future__ import annotations

"""Runtime text logging and structured JSONL event helpers.

This module only handles logging I/O and payload serialization.
It does not decide when events should be emitted.
"""

from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping
import json
import math
import traceback


_VALID_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR"}


@dataclass(frozen=True, slots=True)
class RuntimeLogConfig:
    to_console: bool = True
    to_file: bool = True
    to_jsonl: bool = True
    level: str = "INFO"
    flush: bool = True

    step_summary: bool = True
    outer_detail: bool = True
    inner_summary: bool = True
    inner_monitor: bool = False
    inner_monitor_stride: int = 1

    failure_traceback: bool = True
    copy_tail_to_failure: bool = True


@dataclass(slots=True)
class RuntimeLogger:
    text_path: Path
    jsonl_path: Path | None
    rank: int
    is_root: bool
    cfg: RuntimeLogConfig


def _utc_now_text() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _normalize_level(level: str) -> str:
    token = str(level).strip().upper()
    if token not in _VALID_LEVELS:
        raise ValueError(f"invalid log level: {level!r}")
    return token


def _level_order(level: str) -> int:
    norm = _normalize_level(level)
    return {"DEBUG": 10, "INFO": 20, "WARNING": 30, "ERROR": 40}[norm]


def _should_emit(logger: RuntimeLogger, level: str) -> bool:
    return _level_order(level) >= _level_order(logger.cfg.level)


def _bool_cfg(cfg_section: object, name: str, default: bool) -> bool:
    value = getattr(cfg_section, name, default)
    if isinstance(value, str):
        return value.strip().lower() not in {"0", "false", "no", "off"}
    return bool(value)


def _int_cfg(cfg_section: object, name: str, default: int) -> int:
    try:
        return int(getattr(cfg_section, name, default))
    except Exception:
        return int(default)


def _str_cfg(cfg_section: object, name: str, default: str) -> str:
    return str(getattr(cfg_section, name, default))


def build_runtime_log_config(cfg) -> RuntimeLogConfig:
    diagnostics = getattr(cfg, "diagnostics", None)
    if diagnostics is None:
        return RuntimeLogConfig()

    return RuntimeLogConfig(
        to_console=_bool_cfg(diagnostics, "runtime_log_to_console", True),
        to_file=_bool_cfg(diagnostics, "runtime_log_to_file", True),
        to_jsonl=_bool_cfg(diagnostics, "runtime_log_jsonl", True),
        level=_normalize_level(_str_cfg(diagnostics, "runtime_log_level", "INFO")),
        flush=_bool_cfg(diagnostics, "runtime_log_flush", True),
        step_summary=_bool_cfg(diagnostics, "runtime_log_step_summary", True),
        outer_detail=_bool_cfg(diagnostics, "runtime_log_outer_detail", True),
        inner_summary=_bool_cfg(diagnostics, "runtime_log_inner_summary", True),
        inner_monitor=_bool_cfg(diagnostics, "runtime_log_inner_monitor", False),
        inner_monitor_stride=max(1, _int_cfg(diagnostics, "runtime_log_inner_monitor_stride", 1)),
        failure_traceback=_bool_cfg(diagnostics, "runtime_log_failure_traceback", True),
        copy_tail_to_failure=_bool_cfg(diagnostics, "runtime_log_copy_tail_to_failure", True),
    )


def build_runtime_logger(
    *,
    output_layout,
    cfg,
    rank: int,
    is_root: bool,
) -> RuntimeLogger:
    log_cfg = build_runtime_log_config(cfg)
    jsonl_path = output_layout.run_events_jsonl if log_cfg.to_jsonl else None
    return RuntimeLogger(
        text_path=output_layout.run_log,
        jsonl_path=jsonl_path,
        rank=int(rank),
        is_root=bool(is_root),
        cfg=log_cfg,
    )


def _safe_float(value: Any) -> Any:
    try:
        f = float(value)
    except Exception:
        return value
    if math.isnan(f):
        return "nan"
    if math.isinf(f):
        return "inf" if f > 0.0 else "-inf"
    return f


def _serialize_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return _safe_float(value)
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return _serialize_value(asdict(value))
    if hasattr(value, "_asdict"):
        return _serialize_value(value._asdict())
    if isinstance(value, Mapping):
        return {str(k): _serialize_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_serialize_value(v) for v in value]
    if hasattr(value, "tolist") and callable(getattr(value, "tolist")):
        try:
            return _serialize_value(value.tolist())
        except Exception:
            pass
    if hasattr(value, "item") and callable(getattr(value, "item")):
        try:
            return _serialize_value(value.item())
        except Exception:
            pass
    if hasattr(value, "__dict__"):
        raw: dict[str, Any] = {}
        for key, val in vars(value).items():
            if str(key).startswith("_"):
                continue
            raw[str(key)] = _serialize_value(val)
        return raw
    return _safe_float(value)


def _format_fields(fields: Mapping[str, Any]) -> str:
    if not fields:
        return ""
    parts: list[str] = []
    for key, value in fields.items():
        rendered = _serialize_value(value)
        if isinstance(rendered, (dict, list)):
            text = json.dumps(rendered, ensure_ascii=False, separators=(",", ":"))
        else:
            text = str(rendered)
        parts.append(f"{key}={text}")
    return " " + " ".join(parts)


def _write_text_line(path: Path, line: str, *, flush: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(line)
        fh.write("\n")
        if flush:
            fh.flush()


def _console_print(line: str, *, flush: bool) -> None:
    print(line, flush=flush)


def log_text(
    logger: RuntimeLogger,
    level: str,
    message: str,
    *,
    step_id: int | None = None,
    outer_iter: int | None = None,
    stage: str | None = None,
    **fields: Any,
) -> None:
    norm_level = _normalize_level(level)
    if not _should_emit(logger, norm_level):
        return

    prefix = f"[{_utc_now_text()}][{norm_level}][rank={int(logger.rank)}]"
    if step_id is not None:
        prefix += f"[step={int(step_id)}]"
    if outer_iter is not None:
        prefix += f"[outer={int(outer_iter)}]"
    if stage:
        prefix += f"[stage={stage}]"
    line = f"{prefix} {message}{_format_fields(fields)}"

    if logger.cfg.to_console and logger.is_root:
        _console_print(line, flush=logger.cfg.flush)
    if logger.cfg.to_file and logger.is_root:
        _write_text_line(logger.text_path, line, flush=logger.cfg.flush)


def log_event(
    logger: RuntimeLogger,
    event: str,
    payload: Mapping[str, Any] | None = None,
    *,
    level: str = "INFO",
) -> None:
    norm_level = _normalize_level(level)
    if not _should_emit(logger, norm_level):
        return
    if not logger.cfg.to_jsonl or logger.jsonl_path is None or not logger.is_root:
        return

    record = {
        "ts": _utc_now_iso(),
        "level": norm_level,
        "rank": int(logger.rank),
        "event": str(event),
        "payload": _serialize_value(payload or {}),
    }
    logger.jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with logger.jsonl_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False))
        fh.write("\n")
        if logger.cfg.flush:
            fh.flush()


def log_exception(
    logger: RuntimeLogger,
    where: str,
    exc: BaseException,
    *,
    step_id: int | None = None,
    outer_iter: int | None = None,
    stage: str | None = None,
    extra: Mapping[str, Any] | None = None,
) -> None:
    payload: dict[str, Any] = {
        "where": str(where),
        "exception_type": exc.__class__.__name__,
        "exception_message": str(exc),
    }
    if extra:
        payload.update(_serialize_value(extra))

    log_text(
        logger,
        "ERROR",
        f"{where}: {exc.__class__.__name__}: {exc}",
        step_id=step_id,
        outer_iter=outer_iter,
        stage=stage,
        **payload,
    )
    log_event(logger, "exception", payload, level="ERROR")

    if logger.cfg.failure_traceback:
        tb_text = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)).rstrip()
        for line in tb_text.splitlines():
            log_text(
                logger,
                "ERROR",
                line,
                step_id=step_id,
                outer_iter=outer_iter,
                stage=stage,
            )


def tail_text_file(path: Path, *, n_lines: int = 200) -> list[str]:
    if n_lines <= 0 or not path.exists():
        return []
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return lines[-int(n_lines):]


def copy_log_tail(src: Path, dst: Path, *, n_lines: int = 200) -> None:
    lines = tail_text_file(src, n_lines=n_lines)
    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("w", encoding="utf-8") as fh:
        for line in lines:
            fh.write(line)
            fh.write("\n")


def tail_jsonl_file(path: Path, *, n_lines: int = 200) -> list[str]:
    return tail_text_file(path, n_lines=n_lines)


def copy_jsonl_tail(src: Path, dst: Path, *, n_lines: int = 200) -> None:
    copy_log_tail(src, dst, n_lines=n_lines)


__all__ = [
    "RuntimeLogConfig",
    "RuntimeLogger",
    "build_runtime_log_config",
    "build_runtime_logger",
    "copy_jsonl_tail",
    "copy_log_tail",
    "log_event",
    "log_exception",
    "log_text",
    "tail_jsonl_file",
    "tail_text_file",
]
