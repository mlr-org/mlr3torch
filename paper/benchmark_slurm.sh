#!/bin/bash
#SBATCH --job-name=mlr3torch-benchmark
#SBATCH --partition=mcml-dgx-a100-40x8
#SBATCH --gres=gpu:1
#SBATCH --qos=mcml
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --output=mlr3torch-benchmark-%j.out

# Run the benchmark script
if ! enroot list | grep -q "^mlr3torch-jss$"; then
  enroot create -n mlr3torch-jss sebffischer+mlr3torch-jss+latest.sqsh
fi

enroot start mlr3torch-jss bash -c "
  cd ~/mlr3torch/paper
  cat .Rprofile
  Rscript -e \"
    source('.Rprofile');
    source('benchmark/benchmark.R')
  \"
"
