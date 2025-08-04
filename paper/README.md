# Reproducing the Results

## Computational Environment

To recreate the computational environment, you can download the docker image
`sebffischer/mlr3torch-jss:latest` from dockerhub.

```bash
enroot import docker://index.docker.io#sebffischer/mlr3torch-jss:latest
```

Next, you can create a docker container from the image:

```bash
enroot create --name mlr3torch-jss sebffischer+mlr3torch-jss+latest.sqsh
```

To start the container, run:

```bash
enroot start \
  --mount <parent-dir-to-mlr3torch>:/mnt/data \
  mlr3torch-jss bash
```

## Running the Benchmark

Note that while the benchmark uses `batchtools` for experiment management, we don't use it for job submission in order to ensure that all GPU and CPU benchmarks respectively are run on the same machine.

### Running locally

To run the benchmarks locally, go into `./paper` (to have the right `.Rprofile`).

To initialize the benchmark experiment, run:

```bash
Rscript benchmark/benchmark.R
```

To start the CPU experiments, run:
Note that it's important to have enough RAM, otherwise the benchmarks will be incomparable.

```bash
Rscript benchmark/run_cpu.R
```

To start the GPU experiments, run:

```bash
Rscript benchmark/run_gpu.R
```


### Running on the cluster

Exemplary slurm scripts are provided via `benchmark_init.sh`, `benchmark_cpu.sh`, and `benchmark_gpu.sh`.
These need to be adapted to the specific cluster and job submission system.

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

### Collecting the Results

Once the benchmark experiments are finished, you can collect the results by running:

```bash
Rscript benchmark/summarize.R
```

This will create the `benchmark/results.rds` file.


### Generating the Plots

Simply run:

```r
Rscript paper/plot_benchmark.R
```


## Running the Paper Code

In the docker container, run the following code from the `./paper` directory.
This requires access to an NVIDIA GPU.

```r
knitr::knit('paper_code.Rmd')
```

We also provide a version of the paper code that runs on CPU only.
There, we set the epochs to 0 everywhere and the device to "cpu".

TODOOOO

```r
knitr::knit('paper_code_cheap.Rmd')
```
