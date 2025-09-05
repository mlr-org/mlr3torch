#!/bin/bash
#SBATCH --job-name=mlr3torch-paper
#SBATCH --partition=mcml-hgx-a100-80x4
#SBATCH --gres=gpu:1
#SBATCH --qos=mcml
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --output=mlr3torch-paper-%j.out

cd /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ru48nas2/
enroot create --force --name mlr3torch-jss sebffischer+mlr3torch-jss+latest.sqsh

enroot start \
  --mount  /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ru48nas2/:/mnt/data \
  mlr3torch-jss bash -c "
  cd /mnt/data/mlr3torch/paper
  conda deactivate
  Rscript -e \"
    source('.Rprofile')
    knitr::knit('paper_code.Rmd')
  \"
"
