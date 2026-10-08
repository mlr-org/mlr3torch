#!/bin/bash
#SBATCH --job-name=mlr3torch-benchmark
#SBATCH --partition=mcml-hgx-a100-80x4
#SBATCH --gres=gpu:4
#SBATCH --qos=mcml
#SBATCH --ntasks=1
#SBATCH --time=48:00:00
#SBATCH --exclusive
#SBATCH --output=mlr3torch-benchmark-%j.out

# The whole node is requested (--exclusive) so that no other jobs interfere with the time measurements;
# the benchmark itself only uses one GPU.
DIR=/dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ru48nas2/mlr3torch-paper-review
cd $DIR

echo "Node: $(hostname)"
nvidia-smi
lscpu

enroot create --force --name mlr3torch-jss-gpu mlr3torch-jss+gpu.sqsh

enroot start \
  --mount $DIR:/mnt/data \
  mlr3torch-jss-gpu bash -c "
  cd /mnt/data/paper
  Rscript benchmark/linux-gpu.R
"
