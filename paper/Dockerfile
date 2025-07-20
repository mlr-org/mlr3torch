ARG R_VERSION=4.5.0
# -------- CUDA-capable base image --------
FROM nvcr.io/nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    LANG=en_US.UTF-8 \
    LC_ALL=en_US.UTF-8 \
    ## Posit Public Package Manager – Ubuntu 22.04 binary repo
    RSPM=https://packagemanager.posit.co/cran/__linux__/jammy/latest

# -------- system packages & R --------
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        wget locales curl software-properties-common && \
    locale-gen en_US.UTF-8


RUN curl -Ls https://github.com/r-lib/rig/releases/download/latest/rig-linux-latest.tar.gz | tar xz -C /usr/local && \
  rig add ${R_VERSION} --without-pak && \
  rm -rf /var/lib/apt/lists/* && \
  rm -rf /tmp/rig && \
  apt-get clean

# -------- install Python and pip packages --------
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        python3-dev && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install \
    torch --index-url https://download.pytorch.org/whl/cu124 \
    numpy

# ---------- R defaults: PPM + binaries-only ----------
COPY .Rprofile /etc/R/Rprofile.site

# -------- install R packages
RUN R -q -e "install.packages('torch'); torch::install_torch()"
RUN R -q -e "install.packages('mlr3mbo')"
RUN R -q -e "install.packages('torchdatasets')"
RUN R -q -e "install.packages('mlr3viz')"
RUN R -q -e "install.packages('batchtools')"
RUN R -q -e "install.packages('here')"
RUN R -q -e "install.packages('reticulate')"
RUN R -q -e "install.packages('cowplot')"
RUN R -q -e "install.packages('remotes')"
RUN R -q -e "remotes::install_github('mlr-org/mlr3torch@53c0262')"

CMD ["bash"]
