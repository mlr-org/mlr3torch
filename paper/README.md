# Reproducing the Results

Note that there is also a brief section on reproducibility in the appendix of the paper, which includes a description of the hardware.

## Computational Environment

In order to reproduce the results, you can either use the provided docker images or recreate the `renv` environment that is described in `paper/renv.lock`.
To work with the renv environment, go into the `paper` directory and start an interactive R session, which will bootstrap the environment.
Then, run:

```r
renv::restore()
```

which will ask you whether you want to proceed installing the missing packages, which you have to confirm.

Afterwards, after restarting R just to be sure, you need to run the command below to install torch:

```{r}
torch::install_torch()
```

We are providing two docker images, one for CPU and one for CUDA GPU that have the same packages from the `renv.lock` file installed.
The images can be downloaded from Zenodo: https://doi.org/10.5281/zenodo.17130368, either via the web interface, or, for example, using wget:

```bash
# Docker images
wget https://zenodo.org/records/17864153/files/IMAGE_CPU.tar.gz
wget https://zenodo.org/records/17864153/files/IMAGE_GPU.tar.gz
```

At the time of writing, the images are also hosted on dockerhub, but this is not a permanent storage:
https://hub.docker.com/repository/docker/sebffischer/mlr3torch-jss/general

The `Dockerfile`s used to create the images are available in the `./paper/envs` directory.

```bash
docker load -i /path/to/IMAGE_CPU.tar.gz
```

To start the CPU docker container, run:

```bash
# from anywhere on your machine
docker run -it --rm -v <parent-dir-to-paper>:/mnt/data/ sebffischer/mlr3torch-jss:cpu
# go into the directory that contains the paper code
cd /mnt/data/paper
```

The CUDA image can be started with the command below, which requires the [nvidia extension](https://docs.nvidia.com/ai-enterprise/deployment/vmware/latest/docker.html).

```bash
docker run -it --gpus all --rm -v ../:/mnt/data sebffischer/mlr3torch-jss:gpu
cd /mnt/data/paper
```

Note that the `.Rprofile` file ensures that when running R programs from the `paper` directory, the renv environment will be used unless the code is run in the docker container, where we are not relying on renv directly.

## Running the Benchmark

While the benchmark uses `batchtools` for experiment definition, we don't use it for job submission in order to ensure that all GPU and CPU benchmarks respectively are run on the same machine.
For running the benchmarks, we strongly recommend using the docker images, because we need both PyTorch and (R-)torch, which can be somewhat tricky to setup, especially when using CUDA.

If you want to run it without the docker image, you need to adjust the `PYTHON_PATH` variable in the benchmarking scripts to the path to your Python installation, ensure that `pytorch` is installed and the `"pytorch"` algorithm in `paper/benchmark/benchmark.R` initializes the correct python environment.
But again, we strongly recommend using the provided docker images for the benchmarks.

You can still reproduce the results that compare (R) `torch` with `mlr3torch` without the python environment.
To do so, you can subset the experiments that are run to not include the `"pytorch"` algorithm.
This has to be done in the benchmarking scripts, e.g. `paper/benchmark/linux-gpu.R`.
We show further down how to run only a subset of the jobs.

Note that the CUDA benchmarks were run on a machine with 80 GB VRAM, so errors are expected if you have less.
To address this, you can filter the jobs to restrict the number of layers or latent dimensions as shown further down.

### Running the Benchmarks

Note that the benchmarks take quite some time, which was required to ensure results with high precision that cover many different configurations.
For running a subset of the configurations, see the next section.
Also note that it's important to have enough RAM, otherwise the benchmarks will be non-comparable.
However, there are many other factors, such as the exact hardware that make it generally difficult to reproduce the runtime results.

To run the benchmarks locally, ensure that you are in the `paper` directory.
There are three scripts:

* `paper/benchmark/linux-gpu.R`, which creates the folder `paper/benchmark/registry-linux-gpu`
* `paper/benchmark/linux-cpu.R`, which creates the folder `paper/benchmark/registry-linux-cpu`
* `paper/benchmark/linux-gpu-optimizer.R`, which creates the folder `paper/benchmark/registry-linux-gpu-optimizer`

**Important**:If one of the folders already exists and you want to re-run the benchmarks, you need to delete or move the folder, otherwise you will get an error.
This is to ensure that the benchmark results are not accidentally overwritten.

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

The postprocessd results are stored in:

* `paper/benchmark/result-linux-gpu.rds`
* `paper/benchmark/result-linux-cpu.rds`
* `paper/benchmark/result-linux-gpu-optimizer.rds`

The scripts can, of course, also be run on different machines. The linux names just emphasizes that the provided results are for the linux docker images.

There are also some exemplary slurm scripts that need to be adapted to the specific cluster and job submission system.

* `paper/benchmark/benchmark_gpu.sh`
* `paper/benchmark/benchmark_gpu_optimizer.sh`

### Running a subset of the Jobs

To run a subset of the jobs, modify the table `tbl` in scripts such as `paper/benchmark/linux-gpu.R` to only include the jobs that you want to run.
For example:

```r
ids = tbl[device == "cpu" & n_layers == 10 & latent == 250 & jit & optimizer == "adamw" & repl == 1, ]$job.id
for (id in sample(ids)) {
  submitJobs(id)
  Sys.sleep(0.1)
}
```

### Generating the Benchmark Plots

For the main benchmark shown in the paper, run the following command from the `paper` directory:

```r
Rscript benchmark/plot_benchmark.R
Rscript benchmark/plot_optimizer.R
```

## Recreating the Paper Code

The file `paper/paper_code.R` contains the code from the paper.

You can reproduce it by running the command below from the `paper` directory:

```r
knitr::spin("paper_code.R")
```

We provide the results of running this in `paper/paper_results`.

The results in the paper are those from the CPU docker image and they were fully reproducible when we re-ran them on the same machine.
There were some minor differences in results when re-running the code on a different machine (macOS with M1 CPU vs Linux with Intel CPU).

The file `paper_code.R` contains some very minor differences to the paper we omitted in the paper for brevity.
It was extracted from the tex manuscript almost fully programmatically but adjusted with the following modifications:

* Time measurements (`Sys.time()`)
* Deactivate knitr caching
* Activating caching for `mlr3torch`
* Changing the `mlr3` logging level to `warn` for cleaner output
* Processing the ROC plot for better readability and saving it as `roc.png`, as well as printing it.
* Adding a `sessionInfo()` call at the end

We also added some additional comments to make it easier to associate the code with the paper.

The results we obtained via `knitr::spin()` are stored in `paper/paper_results/`

### Possible Data Unavailability

The code shown in the paper downloads various datasets from standard resources.
In the unlikely but possible event that these datasets are not available anymore, we include:

1. the cache directory for `torch` (MNIST, ResNet-18) and `mlr3torch` (postprocessed MNIST, Melanoma)
2. the dogs-vs-cats dataset

in the Zenodo data.

If one of the downloads (1) fails, download the `cache.tar.gz` file from zenodo, untar it and put it in the location where the cache is (put the `R` folder of the cache into `/root/.cache/R` and the `torch` folder into `/root/.cache/torch` when using the docker images).

If (2) fails, download `dogs-vs-cats.tar.gz` from Zenodo, untar it and put it into the `paper/data` subdirectory where you are running the `paper_code.R` (so the directory structure is `paper/data/dogs-vs-cats`).

To do this in the Docker image you can, e.g., put the files into the parent directory of the `paper` directory (which will be mounted) and then after starting the container, copy the files into the correct location.
Assuming the unpacked cache files are in `/mnt/data/cache`, you can copy them into the correct location with:

```bash
cp -r /mnt/data/cache/R/mlr3torch /root/.cache/R
cp -r /mnt/data/cache/torch /root/.cache/torch
```

### Other errors

When reproducing the results with `knitr` in the docker container, we sometimes encountered issues with the weight downloads for the ResNet-18 model.
This was not an issue when reproducing without `knitr`.
If you also encounter this, delete the problematic model file (you can determine the torch cache directory via `rappdirs::user_cache_dir("torch")`) and download it by running.

```r
torchvision::model_resnet18(pretrained = TRUE)
```

Then, re-run the paper code.
