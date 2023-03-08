# AEDAT

AEDAT is a fast AEDAT 4 Rust reader.

This project was forked from https://github.com/neuromorphicsystems/aedat, to simply remove the Python hooks and to publish it as a crate. You can use this crate when building bespoke Rust software for processing DVS/DAVIS AEDAT4 files--e.g., networked vision systems or event compressors.

[crates.io page](https://crates.io/crates/aedat)

# Documentation

Refer to the [source project](https://github.com/neuromorphicsystems/aedat) for some documentation, especially [src/lib.rs](https://github.com/neuromorphicsystems/aedat/blob/master/src/lib.rs). Future work will focus on adding proper Cargo docs to the Rust code.

## Release notes
### v1.3.0, 2023-03-08
- [x] Update socketed/TCP connections for dv-gui v1.6. This is a breaking (but good) change, as dv-gui added an IO header to the beginning of each packet.

## Development to-do list
- [ ] Add docs
- [ ] Use buffered file readers, if they prove to be faster
