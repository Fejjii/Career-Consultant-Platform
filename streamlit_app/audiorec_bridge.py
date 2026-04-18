"""Keyed wrapper around the ``streamlit-audiorec`` prebuilt component (MediaRecorder → WAV).

The upstream ``st_audiorec.st_audiorec()`` helper re-declares the component on every call and
cannot pass a ``key=``, which makes it easy to double-process the same recording across reruns.
We declare the component once and remount with an incrementing key after each capture so the
next run starts clean.

The browser bundle lives inside the installed ``st_audiorec`` package; no local npm build.
"""

from __future__ import annotations

import os

import st_audiorec
import streamlit.components.v1 as components

_BUILD_DIR = os.path.join(os.path.dirname(st_audiorec.__file__), "frontend", "build")
_st_audiorec_component = components.declare_component("st_audiorec", path=_BUILD_DIR)


def wav_bytes_from_audiorec_payload(raw: dict) -> bytes:
    """Decode the dict payload returned by the st_audiorec iframe into WAV bytes.

    Keys are chunk indices as strings; values are integer byte values 0-255, reassembled in order.
    """
    pairs = sorted((int(k), int(v)) for k, v in raw["arr"].items())
    return b"".join(int(b).to_bytes(1, "big") for _, b in pairs)


def read_wav_from_audiorec(*, key: str) -> bytes | None:
    """Return recorded WAV bytes, or ``None`` if the user has not submitted a new clip."""
    raw = _st_audiorec_component(key=key, default=None)
    if not isinstance(raw, dict) or "arr" not in raw:
        return None
    return wav_bytes_from_audiorec_payload(raw)
