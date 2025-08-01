#!/bin/bash
#SBATCH --job-name=mlr3torch-benchmark
#SBATCH --partition=lrz-cpu
#SBATCH --qos=cpu
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --mem=24G
#SBATCH --exclusive
#SBATCH --output=mlr3torch-benchmark-%j.out

cd /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ru48nas2/
enroot create --force --name mlr3torch-jss sebffischer+mlr3torch-jss+latest.sqsh

enroot start \
  --mount /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ru48nas2/:/mnt/data \
  mlr3torch-jss bash -c "
  cd /mnt/data/mlr3torch/paper
  Rscript -e \"
    source('benchmark/run_cpu.R')
  \"
"
