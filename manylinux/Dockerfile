FROM quay.io/pypa/manylinux2010_x86_64

ENV PATH /root/.cargo/bin:$PATH
ENV PATH /opt/python/cp36-cp36m/bin/:/opt/python/cp37-cp37m/bin/:/opt/python/cp38-cp38/bin/:/opt/python/cp39-cp39/bin/:$PATH
ENV USER root

RUN curl --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
RUN python3 -m pip install --no-cache-dir maturin
RUN mkdir /io

WORKDIR /io
