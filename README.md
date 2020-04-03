# AEDAT

AEDAT is a fast AEDAT 4 python reader, with a Rust underlying implementation.

# Install

This library requires [Python 3.x](https://www.python.org), x >= 5, and [NumPy](https://numpy.org). This guide assumes that they are installed on your machine.

A Rust compiling toolchain is required during the installation (but can be removed afterwards).

## Linux

```sh
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
git clone https://github.com/neuromorphicsystems/aedat.git
cd aedat
rustup toolchain install nightly
rustup override set nightly
cargo build --release
cp target/release/libaedat.so scripts/aedat.so
```

You can now run the python scripts in the *scripts* directory. If you want to import the libary from another directory, copy *aedat.so* in said directory first.

## macOS

```sh
brew install rustup
rustup-init
git clone https://github.com/neuromorphicsystems/aedat.git
cd aedat
rustup toolchain install nightly
rustup override set nightly
cargo build --release
cp target/release/libaedat.dylib scripts/aedat.so
```

You can now run the python scripts in the *scripts* directory. If you want to import the libary from another directory, copy *aedat.so* in said directory first.

## Windows

1. install rustup (instructions availables at https://www.rust-lang.org/tools/install)
2. clone or download this repository
3. run in PowerShell:
  ```sh
  cd aedat
  rustup toolchain install nightly
  rustup override set nightly
  cargo build --release
  cp target/release/libaedat.dll scripts/aedat.pyd
  ```

# Documentation

The `aedat` library provides a single class: `Decoder`. A decoder object is created by passing a file name to `Decoder`. The file name must be a [path-like object](https://docs.python.org/3/glossary.html#term-path-like-object).

The file *scripts/example.py* shows how to use the decoder object. *scripts/example_with_comments* contains the same code with detailed comments.

Because the lifetime of the file handle is managed by Rust, decoder objects are not compatible with the [with](https://docs.python.org/3/reference/compound_stmts.html#with) statement. To ensure garbage collection, point the decoder variable to something else, for example `None`, when you are done using it:
```py
import aedat

decoder = aedat.Decoder('/path/to/file.aedat')
# do something with decoder
decoder = None
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
