# Internal Notes

## TODOs

* [x] Finalize the review + review comments
* [x] Run a cheaper version of the benchmarks with the new docker images with mlr3torch dev version
* [x] Run claude as a deep review against the paper
* [x] Address claude feedback
* [ ] fact check run by claude again (ony grave mistakes)
* [x] Add the 1-thread CPU results to the appendix and multi-thread in main
* [ ] layout
* [ ] Full read (including references)
* [ ] add mlr3torch_paper to slds-lmu and send to Bernmd


* [ ] upload mlr3torch to CRAN once bernd approves and benchmarks are reasonable
* [ ] build final docker images + run final benchmark
* [ ] confirm reproducbibilty of paper results
* [ ] Update the compute info (hardware description) in the paper

* [ ] update zenodo link in paper
* [ ] 1. Two footnotes link to /dev/ documentation; switch them after the CRAN release (the release URLs return 404 now).
* [ ] check that our mechanism for the cache works (when data is no longer available)
* [ ] Prepare all the additional data etc. (handle Mac weird files, gitignore etc.)
* [ ] resubmit
* [ ] check that benchmark still run with final docker images

## Building and Publishing the Docker Images

The images are hosted on dockerhub as `sebffischer/mlr3torch-jss:cpu` and `sebffischer/mlr3torch-jss:gpu` (see `README.md`).
The build context is the `paper` directory, because the `Dockerfile`s copy `renv.lock`.
The image names are fully qualified (`docker.io/...`), so the commands also work with podman.

Run the commands below from the `paper` directory.

### Build

```bash
docker build --platform linux/amd64 -f envs/Dockerfile.linux.cpu -t docker.io/sebffischer/mlr3torch-jss:cpu .
docker build --platform linux/amd64 -f envs/Dockerfile.linux.gpu -t docker.io/sebffischer/mlr3torch-jss:gpu .
```

Check that the PyTorch version matches the libtorch version of the installed R torch package (e.g. R torch 0.17.0 uses libtorch 2.8.0):

```bash
docker run --rm docker.io/sebffischer/mlr3torch-jss:cpu \
  sh -c "Rscript -e 'print(packageVersion(\"torch\"))'; python3 -c 'import torch; print(torch.__version__)'"
```

### Push to dockerhub

```bash
docker login docker.io -u sebffischer
docker push docker.io/sebffischer/mlr3torch-jss:cpu
docker push docker.io/sebffischer/mlr3torch-jss:gpu
```

### Export for Zenodo

```bash
docker save docker.io/sebffischer/mlr3torch-jss:cpu | gzip > IMAGE_CPU.tar.gz
docker save docker.io/sebffischer/mlr3torch-jss:gpu | gzip > IMAGE_GPU.tar.gz
```
