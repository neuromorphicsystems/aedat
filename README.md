# AEDAT

AEDAT is a fast AEDAT 4 python reader, with a Rust underlying implementation.

Run `pip install aedat` to install it.

# Documentation

The `aedat` library provides a single class: `Decoder`. A decoder object is created by passing a file name to `Decoder`. The file name must be a [path-like object](https://docs.python.org/3/glossary.html#term-path-like-object).

Here's a short example:
```python
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

And the same example with detailed comments:

```python
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

Because the lifetime of the file handle is managed by Rust, decoder objects are not compatible with the [with](https://docs.python.org/3/reference/compound_stmts.html#with) statement. To ensure garbage collection, point the decoder variable to something else, for example `None`, when you are done using it:
```py
import aedat

decoder = aedat.Decoder('/path/to/file.aedat')
# do something with decoder
decoder = None
```

# Install from source

This library requires [Python 3.x](https://www.python.org), x >= 5, and [NumPy](https://numpy.org). This guide assumes that they are installed on your machine.

Note for Windows users: this library requires the x86-64 version of Python. You can download it here: https://www.python.org/downloads/windows/ (the default installer contains the x86 version).

A Rust compiling toolchain is required during the installation (but can be removed afterwards).

## Linux

```sh
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
git clone https://github.com/neuromorphicsystems/aedat.git
cd aedat
rustup toolchain install nightly
rustup override set nightly
cargo build --release
cp target/release/libaedat.so aedat.so
```

You can `import aedat` from python scripts in the same directory as *aedat.so*, which can be placed in any directory.

## macOS

```sh
brew install rustup
rustup-init
git clone https://github.com/neuromorphicsystems/aedat.git
cd aedat
rustup toolchain install nightly
rustup override set nightly
cargo build --release
cp target/release/libaedat.dylib aedat.so
```

You can `import aedat` from python scripts in the same directory as *aedat.so*, which can be placed in any directory.

## Windows

1. install rustup (instructions availables at https://www.rust-lang.org/tools/install)
2. clone or download this repository
3. run in PowerShell from the *aedat* directory:
```sh
rustup toolchain install nightly
rustup override set nightly
cargo build --release
copy .\target\release\aedat.dll .\aedat.pyd
```

You can `import aedat` from python scripts in the same directory as *aedat.pyd*, which can be placed in any directory.

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

# Publish

## Requirements

1. Build the Docker image for Linux builds
```
docker build manylinux -t manylinux
```

2. Install all the Pythons for macOS
```
brew install pyenv
pyenv global 3.5.9 3.6.10 3.7.7 3.8.2
pip install maturin
pip install twine
```

3. Download the base Vagrant box for Windows builds

```sh
vagrant box add gusztavvargadr/windows-10
```

## Build and publish


```sh
rm -rf target
eval "$(pyenv init -)"
maturin publish
docker run --rm -v $(pwd):/io manylinux maturin build --release --strip
cd windows
vagrant up
vagrant destroy -f
cd ..
twine upload --skip-existing target/wheels/*
```
Note: The second `maturin` call (i686 target) during the Windows build compiles the dependencies properly, but fails for the library itself (`error: Unrecognized option: 'toolchain'`). The third `maturin` call completes the compilation.
