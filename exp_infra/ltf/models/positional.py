from __future__ import annotations

from enum import Enum


class PEMode(str, Enum):
    NOPE = "nope"
    ROPE = "rope"
    WPE_ALL = "wpe_all"
    WPE_ONCE = "wpe_once"


def normalize_pe_mode(use_wpe: bool, wpe_mode: str, use_rope: bool) -> PEMode:
    if use_wpe and use_rope:
        raise ValueError("use_wpe and use_rope cannot both be true")
    if use_rope:
        return PEMode.ROPE
    if use_wpe:
        if wpe_mode == "all":
            return PEMode.WPE_ALL
        if wpe_mode == "once":
            return PEMode.WPE_ONCE
        raise ValueError(f"use_wpe=true requires wpe_mode in ('all', 'once'), got {wpe_mode!r}")
    return PEMode.NOPE

