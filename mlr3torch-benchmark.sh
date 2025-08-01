#!/bin/bash
#SBATCH --job-name=mlr3torch-benchmark
#SBATCH --partition=lrz-cpu
#SBATCH --qos=cpu
#SBATCH --ntasks=1
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --exclusive
#SBATCH --output=mlr3torch-benchmark-%j.out

cd /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ru48nas2/

# Create container if it doesn't exist
enroot create --force --name mlr3torch-jss sebffischer+mlr3torch-jss+latest.sqsh

# Start container and run commands
enroot start \
  --bind /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ru48nas2:/mnt/data \
  mlr3torch-jss bash -c "cd /mnt/data/mlr3torch && Rscript -e 'print(sessionInfo())' && Rscript -e 'source(\".Rprofile\"); source(\"benchmark/benchmark.R\")'"
