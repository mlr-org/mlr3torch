#!/bin/bash
#SBATCH --job-name=mlr3torch-benchmark
#SBATCH --partition=lrz-cpu
#SBATCH --qos=cpu
#SBATCH --ntasks=1
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --exclusive
#SBATCH --output=mlr3torch-benchmark-%j.out

enroot create --force --name mlr3torch-jss /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ru48nas2/sebffischer+mlr3torch-jss+latest.sqsh

enroot start mlr3torch-jss bash -c "
  conda deactivate
  Rscript -e \"print(sessionInfo())\"
  cd ~/mlr3torch/paper
  cat .Rprofile
  Rscript -e \"
    source('.Rprofile');
    source('benchmark/benchmark.R')
  \"
"
