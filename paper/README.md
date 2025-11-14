# Reproducing the Results

## Computational Environment

In order to reproduce the results, you can either use the provided docker images or recreate the `renv` environment that is located in `./paper/envs/renv`.

You can recreate the `renv` environment by going into the `./paper/envs/renv` directory and running:

```r
renv::init()
```

We are providing two docker images, one for CPU and one for CUDA GPU, that have the same packages from the `renv.lock` file installed.
The images can be downloaded from Zenodo: https://doi.org/10.5281/zenodo.17130368.
You can, for example, use the [zenodo-client](https://pypi.org/project/zenodo-client/) library to download the images:

```bash
# pip install zenodo-client
export ZENODO_API_TOKEN=<your-token>
# for CPU:
zenodo-client download 17130368 IMAGE_CPU.tar.gz
```

By default, the downloaded files are stored in `~/.data/zenodo`.

At the time of writing, the images are also hosted on dockerhub, but this is not a permanent storage:
https://hub.docker.com/repository/docker/sebffischer/mlr3torch-jss/general

The `Dockerfile`s used to create the images are available in the `paper/envs` directory.

After downloading the images, you can load them into Docker, e.g. via:

```bash
docker load -i IMAGE_CPU.tar.gz
```

When using another container manager such as `enroot`, a workaround is to import the image using `Docker` on a system that has it installed and then push it to a dockerhub repository and then pull it from there using `enroot`, along the lines of:

```bash
enroot import docker://sebffischer/mlr3torch-jss:cpu
enroot create --name mlr3torch-jss:cpu sebffischer+mlr3torch-jss+cpu.sqsh
```

To start the container using `Docker`, run:

```bash
docker run -it --rm -v <parent-dir-to-mlr3torch>:/mnt/data/mlr3torch sebffischer/mlr3torch-jss:cpu
# go into the mlr3torch directory
cd /mnt/data/mlr3torch
```

To start the container using `enroot`, run:

```bash
enroot start \
  --mount <parent-dir-to-mlr3torch>:/mnt/data \
  mlr3torch-jss:cpu bash
# go into the mlr3torch directory
cd /mnt/data/mlr3torch
```

## Running the Benchmark

Note that while the benchmark uses `batchtools` for experiment definition, we don't use it for job submission in order to ensure that all GPU and CPU benchmarks respectively are run on the same machine.

For running the benchmarks, we strongly recommend using the docker images, because we need both PyTorch and (R-)torch, which can be somewhat tricky to setup, especially when using CUDA.

If you want to run it without the docker image, you need to ajust the `PYTHON_PATH` variable in the benchmarking scripts to the path to your Python installation, ensure that `pytorch` is installed and the `"pytorch"` algorithm in `./paper/benchmark/benchmark.R` initializes the correct python environment.
But again, we strongly recommend using the provided docker images for the benchmarks.

You can still reproduce the results that compare (R) `torch` with `mlr3torch` without the python environment.
To do so, you can subset the experiments that are run to not include the `"pytorch"` algorithm.
This has to be done in the benchmarking scripts, e.g. `./paper/benchmark/linux-gpu.R`.
We show further down how to run only a subset of the jobs.

### Running the Benchmarks

Note that it's important to have enough RAM, otherwise the benchmarks will be non-comparable.
However, there are many other factors, such as the exact hardware that make it generally difficult to reproduce the exact same results.

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

To run a subset of the jobs, modify the table `tbl` in scripts such as `./paper/benchmark/linux-gpu.R` to only include the jobs that you want to run.
For example:

```r
ids = tbl[device == "cpu" & n_layers == 10 & latent == 250 & jit & optimizer == "adamw" & repl == 1, ]$job.id
for (id in sample(ids)) {
  submitJobs(id)
  Sys.sleep(0.1)
}
```

### Generating the Benchmark Plots

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

## Recreating the Paper Code

The file `./paper/paper_code.R` contains the code from the paper.

You can reproduce it by running:

```r
knitr::spin("paper_code.R")
```

We provide the results of running this in `./paper/paper_results`.

The results in the paper are those from the CPU docker image and they were fully reproducible when we re-ran them on the same machine.
There were some minor differences in results when re-running the code on a different machine (macOS with M1 CPU vs Linux with Intel CPU).

The file `paper_code.R` contains some very minor differences to the paper we omitted in the paper for brevity:
It was extracted from the tex manuscript fully programmatically but adjusted with the following modifications:

* Time measurements (`Sys.time()`)
* Deactivate knitr caching
* Activating caching for `mlr3torch`
* Changing the `mlr3` logging level to `warn` for cleaner output
* Saving the ROC plot for postprocessing
* Adding a `sessionInfo()` call at the end

The results are stored in `./paper/paper_results/`
The ROC plot is postprocessed using the `roc.R` script, which results in the file `paper/paper_results/roc.png`.

### Possible Data Unavailability

The code shown in the paper downloads various datasets from standard resources.
In the unlikely, but possible event that these datasets are not available anymore, we include:

1. the cache directory for `torch` (MNIST, ResNet-18) and `mlr3torch` (postprocessed MNIST, Melanoma)
2. the dogs-vs-cats dataset

in the Zenodo data.

If one of the downloads (1) fails, download the `cache.tar.gz` file from zenodo, untar it and put it in the location where the cache is (put it as `/root/.cache/` when using the docker images).

If (2) fails, download `dogs-vs-cats.tar.gz` from Zenodo, untar it and put it into the directory where you are running the `paper_code.R`.

### Other errors

When reproducing the results with `knitr` in the docker container, we sometimes encountered issues with the weight downloads for the ResNet-18 model.
This was not an issue when reproducing without `knitr`.
If you also encounter this, delete the problematic model file (you can determine the torch cache directory via `rappdirs::user_cache_dir("torch")`) and download it by running.

```r
torchvision::model_resnet18(pretrained = TRUE)
```

Then, re-run the paper code.
