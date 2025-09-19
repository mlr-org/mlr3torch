# Reproducing the Results

## Computational Environment

For reproducibility, two linux docker images are provided for CPU and CUDA GPU:
<ZENODO LINK>

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

Note that while the benchmark uses `batchtools` for experiment management, we don't use it for job submission in order to ensure that all GPU and CPU benchmarks respectively are run on the same machine.

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

The ROC plot is postprocessed using the `roc.R` script.

## Running the Paper Code

In the docker container, run the following code from the `./paper` directory.
This requires access to an NVIDIA GPU and usage of the CUDA docker image.

```r
knitr::knit('paper_code.Rmd')
```

The result of the above is `paper_code.md`.
The ROC plot is postprocessed using the `roc.R` script.

In order to demonstrate reprodicbility of the code on CPU (see paper Appendix A), we include a considerably simplified version of the paper code, where the tasks are subset to only contain a few rows and some other hyperparameters are adjusted.
This means the results are not meaningful, but it allows to run the code easily on a CPU.
Use the linux docker image for CPU for this.

```r
knitr::knit('paper_code_cheap_cpu.Rmd')
```

The results of running this on the CPU container are included in `paper_code_cheap_cpu.md`
