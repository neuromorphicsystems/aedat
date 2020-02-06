# AEDAT

AEDAT is a fast AEDAT 4 python reader, with a Rust underlying implementation.

# Install

This library requires [Python 3.x](https://www.python.org), x >= 5, and [NumPy](https://numpy.org). This guide assumes that they are installed on your machine.


A Rust compiling toolchain is required during the installation (but can be removed afterwards).

## Linux

```sh
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
git clone https://http://github.com/neuromorphicsystems/aedat.git
cd aedat
rustup toolchain install nightly
rustup override set nightly
cargo build --release
```

After compiling, copy the generated library *target/release/libaedat.so* in the directory of your choice, and rename it *aedat<span>.</span>so*. The python scripts in this directory can now `import aedat`.

## macOS

```sh
brew install rustup
rustup-init
git clone https://http://github.com/neuromorphicsystems/aedat.git
cd aedat
rustup toolchain install nightly
rustup override set nightly
cargo build --release
```

After compiling, copy the generated library *target/release/libaedat.dylib* in the directory of your choice, and rename it *aedat<span>.</span>so*. The python scripts in this directory can now `import aedat`.

# Documentation

The following script decodes the file `/path/to/file.aedat`. Basic information is printed for each packet:

```py
import aedat

decoder = aedat.Decoder('/path/to/file.aedat')
print(decoder.id_to_stream())

for packet in decoder:
    print(packet['stream_id'], end=': ')
    if 'events' in packet:
        print('{} polarity events'.format(len(packet['events'])))
    elif 'frame' in packet:
        print('{} x {} frame'.format(packet['frame']['width'], packet['frame']['height']))
    elif 'imus' in packet:
        print('{} IMU samples'.format(len(packet['imus'])))
    elif 'triggers' in packet:
        print('{} trigger events'.format(len(packet['triggers'])))
```

The following code block contains the same script, but adds detailed documentation on the content of each packet:

```py
import aedat

decoder = aedat.Decoder('/path/to/file.aedat')
"""
decoder is a packet iterator with an additional method id_to_stream
id_to_stream returns a dictionary with the following structure:
{
    <int>: {
        'type': <str>,
    }
}
type is one of 'events', 'frame', 'imus', 'triggers'
if type is 'events' or 'frame', its parent dictionary has the following structure:
{
    'type': <str>,
    'width': <int>,
    'height': <int>,
}
"""
print(decoder.id_to_stream())

for packet in decoder:
    """
    packet is a dictionary with the following structure:
    {
        'stream_id': <int>,
    }
    packet also has exactly one of the following fields:
        'events', 'frame', 'imus', 'triggers'
    """
    print(packet['stream_id'], end=': ')
    if 'events' in packet:
        """
        packet['events'] is a structured numpy array with the following dtype:
            [
                ('t', '<u8'),
                ('x', '<u2'),
                ('y', '<u2'),
                ('on', '?'),
            ]
        """
        print('{} polarity events'.format(len(packet['events'])))
    elif 'frame' in packet:
        """
        packet['frame'] is a dictionary with the following structure:
            {
                't': <int>,
                'begin_t': <int>,
                'end_t': <int>,
                'exposure_begin_t': <int>,
                'exposure_end_t': <int>,
                'format': <str>,
                'width': <int>,
                'height': <int>,
                'offset_x': <int>,
                'offset_y': <int>,
                'pixels': <numpy.array(shape=(height, width), dtype=uint8)>,
            }
        format is one of 'Gray', 'BGR', 'BGRA'
        """
        print('{} x {} frame'.format(packet['frame']['width'], packet['frame']['height']))
    elif 'imus' in packet:
        """
        packet['imus'] is a structured numpy array with the following dtype:
            [
                ('t', '<u8'),
                ('temperature', '<f4'),
                ('accelerometer_x', '<f4'),
                ('accelerometer_y', '<f4'),
                ('accelerometer_z', '<f4'),
                ('gyroscope_x', '<f4'),
                ('gyroscope_y', '<f4'),
                ('gyroscope_z', '<f4'),
                ('magnetometer_x', '<f4'),
                ('magnetometer_y', '<f4'),
                ('magnetometer_z', '<f4'),
            ]
        """
        print('{} IMU samples'.format(len(packet['imus'])))
    elif 'triggers' in packet:
        """
        packet['triggers'] is a structured numpy array with the following dtype:
            [
                ('t', '<u8'),
                ('source', 'u1'),
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
        print('{} trigger events'.format(len(packet['triggers'])))
```


# Contribute

After changing any of the files in *framebuffers*, one must run:
```sh
flatc --rust -o src/ flatbuffers/*.fbs
```

To format the code, run:
```sh
cargo fmt
```
You may need to install rustfmt first with:
```sh
rustup component add rustfmt
```
