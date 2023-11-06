AEDAT is a fast AEDAT 4 python reader, with a Rust underlying implementation.

Run `pip install aedat` to install it.

- [Example](#example)
- [Process frames](#process-frames)
    - [Pillow (PIL)](#pillow-pil)
    - [OpenCV](#opencv)
- [Detailed example](#detailed-example)
- [Install from source](#install-from-source)
    - [Linux](#linux)
    - [macOS](#macos)
    - [Windows](#windows)
- [Contribute](#contribute)
- [Publish](#publish)

## Example

The `aedat` library provides a single class: `Decoder`. A decoder object is created by passing a file name to `Decoder`. The file name must be a [path-like object](https://docs.python.org/3/glossary.html#term-path-like-object).

```python
import aedat

decoder = aedat.Decoder("/path/to/file.aedat")
print(decoder.id_to_stream())

for packet in decoder:
    print(packet["stream_id"], end=": ")
    if "events" in packet:
        print("{} polarity events".format(len(packet["events"])))
    elif "frame" in packet:
        print("{} x {} frame".format(packet["frame"]["width"], packet["frame"]["height"]))
    elif "imus" in packet:
        print("{} IMU samples".format(len(packet["imus"])))
    elif "triggers" in packet:
        print("{} trigger events".format(len(packet["triggers"])))
```

## Process frames

### Pillow (PIL)

```py
import aedat
import PIL.Image # https://pypi.org/project/Pillow/

index = 0
for packet in decoder:
    if "frame" in packet:
        image = PIL.Image.fromarray(
            packet["frame"]["pixels"],
            mode=packet["frame"]["format"],
        )
        image.save(f"{index}.png")
        index += 1
```

### OpenCV

```py
import aedat
import cv2 # https://pypi.org/project/opencv-python/

index = 0
for packet in decoder:
    if "frame" in packet:
        image = packet["frame"]["pixels"]
        if packet["frame"]["format"] == "RGB":
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        elif packet["frame"]["format"] == "RGBA":
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGRA)
        cv2.imwrite(f"{index}.png", image)
        index += 1
```

## Detailed example

This is the same as the first example, with detailed comments:

```python
import aedat

decoder = aedat.Decoder("/path/to/file.aedat")
"""
decoder is a packet iterator with an additional method id_to_stream
id_to_stream returns a dictionary with the following structure:
{
    <int>: {
        "type": <str>,
    }
}
type is one of "events", "frame", "imus", "triggers"
if type is "events" or "frame", its parent dictionary has the following structure:
{
    "type": <str>,
    "width": <int>,
    "height": <int>,
}
"""
print(decoder.id_to_stream())

for packet in decoder:
    """
    packet is a dictionary with the following structure:
    {
        "stream_id": <int>,
    }
    packet also has exactly one of the following fields:
        "events", "frame", "imus", "triggers"
    """
    print(packet["stream_id"], end=": ")
    if "events" in packet:
        """
        packet["events"] is a structured numpy array with the following dtype:
            [
                ("t", "<u8"),
                ("x", "<u2"),
                ("y", "<u2"),
                ("on", "?"),
            ]
        """
        print("{} polarity events".format(len(packet["events"])))
    elif "frame" in packet:
        """
        packet["frame"] is a dictionary with the following structure:
            {
                "t": <int>,
                "begin_t": <int>,
                "end_t": <int>,
                "exposure_begin_t": <int>,
                "exposure_end_t": <int>,
                "format": <str>,
                "width": <int>,
                "height": <int>,
                "offset_x": <int>,
                "offset_y": <int>,
                "pixels": <numpy.array(shape=(height, width), dtype=uint8)>,
            }
        format is one of "L", "RGB", "RGBA"
        """
        print("{} x {} frame".format(packet["frame"]["width"], packet["frame"]["height"]))
    elif "imus" in packet:
        """
        packet["imus"] is a structured numpy array with the following dtype:
            [
                ("t", "<u8"),
                ("temperature", "<f4"),
                ("accelerometer_x", "<f4"),
                ("accelerometer_y", "<f4"),
                ("accelerometer_z", "<f4"),
                ("gyroscope_x", "<f4"),
                ("gyroscope_y", "<f4"),
                ("gyroscope_z", "<f4"),
                ("magnetometer_x", "<f4"),
                ("magnetometer_y", "<f4"),
                ("magnetometer_z", "<f4"),
            ]
        """
        print("{} IMU samples".format(len(packet["imus"])))
    elif "triggers" in packet:
        """
        packet["triggers"] is a structured numpy array with the following dtype:
            [
                ("t", "<u8"),
                ("source", "u1"),
            ]
        the source value has the following meaning:
            0: timestamp reset
            1: external signal rising edge
            2: external signal falling edge
            3: external signal pulse
            4: external generator rising edge
            5: external generator falling edge
            6: frame begin
            7: frame end
            8: exposure begin
            9: exposure end
        """
        print("{} trigger events".format(len(packet["triggers"])))
```

Because the lifetime of the file handle is managed by Rust, decoder objects are not compatible with the [with](https://docs.python.org/3/reference/compound_stmts.html#with) statement. To ensure garbage collection, point the decoder variable to something else, for example `None`, when you are done using it:

```py
import aedat

decoder = aedat.Decoder("/path/to/file.aedat")
# do something with decoder
decoder = None
```

## Install from source

Local build (first run).

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install maturin numpy
maturin develop  # or maturin develop --release to build with optimizations
```

Local build (subsequent runs).

```sh
source .venv/bin/activate
maturin develop  # or maturin develop --release to build with optimizations
```

After changing any of the files in _framebuffers_, one must run:

```sh
flatc --rust -o src/ flatbuffers/*.fbs
```

To format the code, run:

```sh
cargo fmt
```

# Publish

1. Bump the version number in _Cargo.toml_ and _pyproject.toml_.

2. Create a new release on GitHub.
