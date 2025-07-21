#!/bin/bash
#SBATCH --job-name=mlr3torch-benchmark
#SBATCH --partition=mcml-hgx-a100-80x4
#SBATCH --gres=gpu:4
#SBATCH --qos=mcml
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --exclusive
#SBATCH --output=mlr3torch-paper-%j.out

# Run the benchmark script
if ! enroot list | grep -q "^mlr3torch-jss2$"; then
  enroot create -n mlr3torch-jss2 sebffischer+mlr3torch-jss+latest.sqsh
fi

enroot start mlr3torch-jss2 bash -c "
  cd ~/mlr3torch/paper
  Rscript -e \"
    source('.Rprofile')
    knitr::knit('paper_code.Rmd')
  \"
"
