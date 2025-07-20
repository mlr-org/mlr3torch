# Reproducing the Results

 ## Computational Environment

To recreate the computational environment, you can download docker container
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
enroot start mlr3torch-jss
```

To start the benchmark experiment, go into `./paper` and run:

```bash
Rscript benchmark/benchmark.R
```

If you want to only run a subset of the experiments, you
