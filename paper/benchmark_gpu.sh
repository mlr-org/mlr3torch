#!/bin/bash
#SBATCH --job-name=mlr3torch-benchmark
#SBATCH --partition=mcml-hgx-a100-80x4
#SBATCH --gres=gpu:4
#SBATCH --qos=mcml
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --exclusive
#SBATCH --output=mlr3torch-benchmark-%j.out

enroot create --force --name mlr3torch-jss /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ru48nas2/sebffischer+mlr3torch-jss+latest.sqsh

enroot start mlr3torch-jss bash -c "
  cd ~/mlr3torch/paper
  Rscript -e \"
    source('.Rprofile');
    source('benchmark/run_gpu.R')
  \"
"
