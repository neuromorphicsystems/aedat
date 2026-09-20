from __future__ import annotations

import pathlib
import typing

import numpy
import pytest

import aedat

data = pathlib.Path(__file__).resolve().parent / "data"


def decode_frames(path: pathlib.Path) -> list[dict[str, typing.Any]]:
    return [packet["frame"] for packet in aedat.Decoder(path) if "frame" in packet]


def test_decode_gray16():
    frames = decode_frames(data / "test_data_gray16.aedat4")
    assert len(frames) == 1
    assert frames[0]["format"] == "I;16"
    assert frames[0]["width"] == 2
    assert frames[0]["height"] == 2
    assert frames[0]["pixels"].dtype == numpy.uint16
    assert frames[0]["pixels"].tolist() == [[1, 2], [3, 1023]]


def test_unknown_format_does_not_hide_later_frames():
    with pytest.warns(UserWarning, match="unknown frame format 1"):
        frames = decode_frames(data / "test_data_unknown_then_gray16.aedat4")
    assert [frame["format"] for frame in frames] == ["I;16"]
    assert frames[0]["pixels"][1, 1] == 1023
