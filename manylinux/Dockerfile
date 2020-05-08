FROM quay.io/pypa/manylinux1_x86_64

ENV PATH /root/.cargo/bin:$PATH
ENV PATH /opt/python/cp35-cp35m/bin/:/opt/python/cp36-cp36m/bin/:/opt/python/cp37-cp37m/bin/:/opt/python/cp38-cp38/bin/:$PATH
ENV USER root

RUN curl https://sh.rustup.rs -sSf | sh -s -- -y \
    && rustup toolchain add nightly-2020-01-23 \
    && rustup default nightly-2020-01-23
RUN python3 -m pip install --no-cache-dir maturin
RUN mkdir /io

WORKDIR /io
