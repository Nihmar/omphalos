"""Helpers to read gguf-py metadata fields as plain Python values."""

from __future__ import annotations

import numpy as np


def field_value(field):
    try:
        value = field.contents()
    except Exception:
        return None
    if isinstance(value, np.ndarray):
        if value.dtype.kind in "OSU":
            out = []
            for item in value.tolist():
                if isinstance(item, bytes):
                    item = item.decode("utf-8", "replace")
                out.append(item)
            return out
        return value.tolist()
    return value


def scalar(fields, key, default=None):
    value = field_value(fields[key]) if key in fields else None
    if value is None:
        return default
    if isinstance(value, list) and len(value) == 1:
        return value[0]
    return value


def text(fields, key, default=""):
    value = scalar(fields, key)
    if value is None:
        return default
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    if isinstance(value, list):
        return " ".join(str(v) for v in value)
    return str(value)


def get_int(fields, key, default):
    value = scalar(fields, key)
    return int(value) if value is not None else default


def get_float(fields, key, default):
    value = scalar(fields, key)
    return float(value) if value is not None else default


def get_list(fields, key, default):
    value = scalar(fields, key)
    return value if isinstance(value, list) else default
