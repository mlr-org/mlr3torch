# Reproducing the Results

## Computational Environment

For reproducibility, two linux docker images are provided for CPU and CUDA GPU, see the Zenodo link in the appendix of the paper.

You can, e.g., download the images via the [zenodo-client](https://pypi.org/project/zenodo-client/) library:

```bash
# pip install zenodo-client
export ZENODO_API_TOKEN=<your-token>
# for CPU:
zenodo-client download 17140855 IMAGE_CPU.tar.gz
```

By default, the downloaded files are stored in `~/.data/zenodo`.

At the time of writing, the images are also hosted on dockerhub, but this is not a permanent storage:
https://hub.docker.com/repository/docker/sebffischer/mlr3torch-jss/general

The `Dockerfile`s used to create the images are available in the `paper/envs` directory.

After downloading the images, you can load them into Docker, e.g. via:

```bash
docker load -i IMAGE_CPU.tar.gz
```

When using another container manager such as `enroot`, a workaround is to import the image using `Docker` on a system that has it installed and then push it to a dockerhub repository and then pull it from there using `enroot`, via:

```bash
enroot import docker://sebffischer/mlr3torch-jss:cpu
enroot create --name mlr3torch-jss:cpu sebffischer+mlr3torch-jss+cpu.sqsh
```

To start the container using `Docker`, run:

```bash
docker run -it --rm -v <parent-dir-to-mlr3torch>:/mnt/data/mlr3torch sebffischer/mlr3torch-jss:cpu
```

To start the container using `enroot`, run:

```bash
enroot start \
  --mount <parent-dir-to-mlr3torch>:/mnt/data \
  mlr3torch-jss:cpu bash
```

Some code expects this directory structure, so make sure to mount the directory like above.

## Running the Benchmark

Note that while the benchmark uses `batchtools` for experiment definition, we don't use it for job submission in order to ensure that all GPU and CPU benchmarks respectively are run on the same machine.

### Running locally

Note that it's important to have enough RAM, otherwise the benchmarks will be incomparable.

To run the benchmarks locally, go into `./paper`:

To run the GPU benchmarks (using the CUDA docker image) on linux, run:

```bash
Rscript benchmark/linux-gpu.R
```

To run the CPU benchmarks (using the CPU docker image) on linux, run:

```bash
Rscript benchmark/linux-cpu.R
```

To run the benchmark that compares "ignite" with standard optimizers (using the CUDA docker image) on linux, run:

```bash
Rscript benchmark/linux-gpu-optimizer.R
```

The results are stored in:

* `paper/benchmark/result-linux-gpu.rds`
* `paper/benchmark/result-linux-cpu.rds`
* `paper/benchmark/result-linux-gpu-optimizer.rds`

There are also some exemplary slurm scripts that need to be adapted to the specific cluster and job submission system.

* `paper/benchmark/benchmark_gpu.sh`
* `paper/benchmark/benchmark_gpu_optimizer.sh`

### Running a subset of the Jobs

To run a subset of the jobs, you can adjust the runner scripts to do something along the lines of:

```r
reg = loadRegistry("~/mlr3torch/paper/benchmark/registry", writeable = TRUE)
tbl = unwrap(getJobTable(reg))
ids = tbl[device == "cpu" & n_layers == 10 & latent == 250 & jit & optimizer == "adamw" & repl == 1, ]$job.id
for (id in sample(ids)) {
  submitJobs(id)
  Sys.sleep(0.1)
}
```

### Generating the Plots

For the main benchmark shown in the paper, run:

```r
Rscript paper/benchmark/plot_benchmark.R
```

For the comparison of "ignite" with standard optimizers, run:

```r
Rscript paper/benchmark/plot_optimizer.R
```

These commands generate the files:

* `paper/benchmark/plot_benchmark.png`
* `paper/benchmark/plot_benchmark_relative.png`
* `paper/benchmark/plot_optimizer.png`

## Running the Paper Code

In the docker container, run the following code from the `./paper` directory.
This requires access to an NVIDIA GPU and usage of the CUDA docker image.

```r
knitr::knit('paper_code.Rmd')
```

Note that this Rmd file was extracted from the tex source using `extract.R`.
However, we added some minor modifications, which includes:

* Activating caching for `mlr3torch`
* Changing the `mlr3` logging level to `warn` for cleaner output
* Saving the ROC plot for postprocessing
* Adding a `sessionInfo()` call at the end

The result of the above is `paper_code.md`.
The ROC plot is postprocessed using the `roc.R` script and saved in `paper/roc.rds`.

In order to demonstrate reprodicbility of the code on CPU (see paper Appendix A), we include a considerably simplified version of the paper code, where the tasks are subset to only contain a few rows and some other hyperparameters are adjusted.
This means the results are "worse" and less realistic, but it allows to run the code easily on a CPU in a reasonable amount of time.
Use the linux Docker image for CPU for this.

```r
knitr::knit('paper_code_cheap_cpu.Rmd')
```

The results of running this on the CPU container are included in `paper_code_cheap_cpu.md`

### Possible Data Unavailability

The code shown in the paper downloads various datasets from standard resources.
In the unlikely, but possible event that these datasets are not available anymore, we include:

1. the cache directory for `torch` (MNIST, ResNet-18) and `mlr3torch` (postprocessed MNIST, Melanoma)
2. the dogs-vs-cats dataset

in the Zenodo data.

If one of the downloads (1) fails, download the `cache.tar.gz` file from zenodo, untar it and put it in the location where the cache is (put it as `/root/.cache/` when using the docker images).

If (2) fails, download `dogs-vs-cats.tar.gz` from Zenodo, untar it and put it into `./paper/data`.

### Other errors

When reproducing the results with `knitr` in the docker container, we sometimes encountered issues with the weight downloads for the ResNet-18 model.
If you also encounter this, delete the problematic model file and download it by running

```r
torchvision::model_resnet18(pretrained = TRUE)
```
