FROM rocker/r-ver:4.3.1

ENV MKLROOT=/opt/intel/oneapi/mkl/latest
RUN ls $MKLROOT/lib/intel64

ENV MKLROOT=/opt/intel/oneapi/mkl/latest
ENV MKL_LINKLINE="-L$MKLROOT/lib/intel64 -lmkl_rt -lpthread -lm -ldl"

RUN apt-get update && apt-get install -y \
    wget \
    build-essential \
    libcurl4-openssl-dev \
    libssl-dev \
    libxml2-dev

RUN wget -q https://cran.r-project.org/src/base/R-4/R-4.5.0.tar.gz -O /tmp/R-4.5.0.tar.gz && \
    tar -xzf /tmp/R-4.5.0.tar.gz -C /tmp && rm /tmp/R-4.5.0.tar.gz && \
    cd /tmp/R-4.5.0 && \
    ./configure --with-blas="$MKL_LINKLINE" --with-lapack="$MKL_LINKLINE" && \
    make -j"$(nproc)" && make install && \
    cd / && rm -rf /tmp/R-4.5.0

RUN grep -n "BLAS" /tmp/R-4.5.0/config.log || true
