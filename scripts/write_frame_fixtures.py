"""Write tiny uncompressed AEDAT-4 files used by test.py (stdlib + flatbuffers)."""

from pathlib import Path

import flatbuffers

MAGIC = b"#!AER-DAT4.0\r\n"
DESCRIPTION = """<dv version="2.0">
    <node name="outInfo">
        <node name="0">
            <attr key="typeIdentifier" type="string">FRME</attr>
            <node name="info">
                <attr key="sizeX" type="int">2</attr>
                <attr key="sizeY" type="int">2</attr>
            </node>
        </node>
    </node>
</dv>
"""


def ioheader() -> bytes:
    builder = flatbuffers.Builder(256)
    description = builder.CreateString(DESCRIPTION)
    builder.StartObject(3)
    builder.PrependUOffsetTRelativeSlot(2, description, 0)
    root = builder.EndObject()
    builder.FinishSizePrefixed(root, b"IOHE")
    return bytes(builder.Output())


def frame_packet(fmt: int, pixel_bytes: bytes, width: int = 2, height: int = 2) -> bytes:
    builder = flatbuffers.Builder(256)
    pixels = builder.CreateByteVector(pixel_bytes)
    builder.StartObject(11)
    builder.PrependUOffsetTRelativeSlot(10, pixels, 0)
    builder.PrependInt16Slot(9, 0, 0)
    builder.PrependInt16Slot(8, 0, 0)
    builder.PrependInt16Slot(7, height, 0)
    builder.PrependInt16Slot(6, width, 0)
    builder.PrependInt8Slot(5, fmt, 0)
    builder.PrependInt64Slot(4, 1050, 0)
    builder.PrependInt64Slot(3, 950, 0)
    builder.PrependInt64Slot(2, 1100, 0)
    builder.PrependInt64Slot(1, 900, 0)
    builder.PrependInt64Slot(0, 1000, 0)
    root = builder.EndObject()
    builder.FinishSizePrefixed(root, b"FRME")
    payload = bytes(builder.Output())
    return (0).to_bytes(4, "little") + len(payload).to_bytes(4, "little") + payload


def write_aedat4(path: Path, packets: list[bytes]) -> None:
    path.write_bytes(MAGIC + ioheader() + b"".join(packets))
    print(f"wrote {path} ({path.stat().st_size} bytes)")


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    gray16_pixels = bytes(
        [
            1,
            0,
            2,
            0,
            3,
            0,
            0xFF,
            0x03,
        ]
    )
    write_aedat4(root / "test_data_gray16.aedat4", [frame_packet(2, gray16_pixels)])
    unknown_then_gray16 = [
        frame_packet(1, bytes([10, 20, 30, 40])),
        frame_packet(2, gray16_pixels),
    ]
    write_aedat4(root / "test_data_unknown_then_gray16.aedat4", unknown_then_gray16)


if __name__ == "__main__":
    main()
