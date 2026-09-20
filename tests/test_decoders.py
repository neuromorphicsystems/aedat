from __future__ import annotations

import gc
import pathlib
import struct
import sys

import numpy
import pytest

import aedat

from . import assets


def little_endian_bytes(array: numpy.ndarray) -> bytes:
    return array.astype(array.dtype.newbyteorder("<"), copy=False).tobytes()


def count_packets(path: pathlib.Path) -> int:
    return sum(1 for _ in aedat.Decoder(path))


@pytest.mark.parametrize("file", assets.files)
def test_decoder(file: assets.File):
    decoder = aedat.Decoder(file.path)
    assert decoder.id_to_stream() == file.id_to_stream
    field_to_hasher = file.field_to_hasher()
    for packet in decoder:
        if "events" in packet:
            events = packet["events"]
            assert events.dtype.names == ("t", "x", "y", "on")
            assert numpy.array_equal(events["on"], events["p"])
            field_to_hasher["t"].update(little_endian_bytes(events["t"]))
            field_to_hasher["x"].update(little_endian_bytes(events["x"]))
            field_to_hasher["y"].update(little_endian_bytes(events["y"]))
            field_to_hasher["on"].update(little_endian_bytes(events["on"]))
        elif "frame" in packet:
            field_to_hasher["frame"].update(
                little_endian_bytes(packet["frame"]["pixels"])
            )
        elif "imus" in packet:
            field_to_hasher["imus"].update(little_endian_bytes(packet["imus"]))
        elif "triggers" in packet:
            field_to_hasher["triggers"].update(little_endian_bytes(packet["triggers"]))
        else:
            pytest.fail(f"unexpected packet {sorted(packet.keys())}")
    for field, hasher in field_to_hasher.items():
        assert hasher.hexdigest() == file.field_to_digest[field], f"{file=}, {field=}"


@pytest.mark.parametrize("file", assets.files)
def test_decoder_does_not_leak(file: assets.File):
    count_packets(file.path)
    gc.collect()
    blocks_before = sys.getallocatedblocks()
    packets = count_packets(file.path)
    gc.collect()
    assert sys.getallocatedblocks() - blocks_before < packets


@pytest.mark.parametrize("file", assets.files)
def test_truncated_file_raises(file: assets.File, tmp_path: pathlib.Path):
    raw = file.path.read_bytes()
    offset = len("#!AER-DAT4.0\r\n") + 4 + struct.unpack("<I", raw[14:18])[0]
    for _ in range(20):
        offset += 8 + struct.unpack("<I", raw[offset + 4 : offset + 8])[0]
    truncated = tmp_path / "truncated.aedat4"
    truncated.write_bytes(raw[:offset])
    with pytest.raises(RuntimeError):
        count_packets(truncated)
