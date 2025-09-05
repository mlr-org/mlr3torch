#!/bin/bash
#SBATCH --job-name=mlr3torch-benchmark
#SBATCH --partition=mcml-hgx-a100-80x4
#SBATCH --gres=gpu:4
#SBATCH --qos=mcml
#SBATCH --ntasks=1
#SBATCH --time=48:00:00
#SBATCH --exclusive
#SBATCH --output=mlr3torch-benchmark-%j.out

cd /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ru48nas2/
enroot create --force --name mlr3torch-jss sebffischer+mlr3torch-jss+gpu.sqsh

enroot start \
  --mount /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ru48nas2/:/mnt/data \
  mlr3torch-jss bash -c "
  cd /mnt/data/mlr3torch/paper
  Rscript -e \"
    source('benchmark/linux-gpu-optimizer.R')
  \"
"
