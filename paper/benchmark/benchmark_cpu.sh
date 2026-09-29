#!/bin/bash
#SBATCH --job-name=mlr3torch-benchmark-cpu
#SBATCH --partition=lrz-cpu
#SBATCH --qos=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=48:00:00
#SBATCH --output=mlr3torch-benchmark-cpu-%j.out

# The whole node is requested (--exclusive) so that no other jobs interfere with the time measurements.
# The benchmark itself sets the number of threads (1 or 16) via torch_set_num_threads().

DIR=/dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ru48nas2/mlr3torch-paper-review
cd $DIR

echo "Node: $(hostname)"
lscpu
free -g

enroot create --force --name mlr3torch-jss-cpu mlr3torch-jss+cpu.sqsh

enroot start \
  --mount $DIR:/mnt/data \
  mlr3torch-jss-cpu bash -c "
  cd /mnt/data/paper
  Rscript benchmark/linux-cpu.R
"
