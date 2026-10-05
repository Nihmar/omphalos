"""The .omph source digest (#346): the header records the GGUF it came from, and
omph_file() refuses a stale conversion."""

from __future__ import annotations

import hashlib
import struct

import pytest

import omph_model


def make_omph(path, digest: str, extra_bytes: bytes = b"") -> None:
    def s(x: bytes) -> bytes:
        return struct.pack("<Q", len(x)) + x

    data = b"OMPH" + struct.pack("<I", 3) + struct.pack("<Q", 0) + struct.pack("<Q", 1)
    data += s(b"omph.source_sha256") + struct.pack("<I", 8) + s(digest.encode())
    data += extra_bytes
    path.write_bytes(data)


def test_source_digest(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(omph_model, "_CACHE", tmp_path / "cache.json")
    monkeypatch.setattr(omph_model, "_MEM", {})
    gguf = tmp_path / "m.gguf"
    gguf.write_bytes(b"hello")
    digest = hashlib.sha256(b"hello").hexdigest()
    omph = tmp_path / "m.omph"
    make_omph(omph, digest)
    assert omph_model.source_sha256(omph) == digest
    assert omph_model.omph_file(str(gguf)) == str(omph)
    # the cache: a second call does not rehash (the size and mtime are the key)
    assert omph_model.file_sha256(gguf) == digest
    # a GGUF replaced in place: same name, different bytes -> refuse
    gguf.write_bytes(b"other bytes, a different size")
    with pytest.raises(SystemExit):
        omph_model.omph_file(str(gguf))


def test_no_digest_key(tmp_path) -> None:
    omph = tmp_path / "old.omph"
    omph.write_bytes(b"OMPH" + struct.pack("<I", 3) + struct.pack("<Q", 0) + struct.pack("<Q", 0))
    assert omph_model.source_sha256(omph) is None
