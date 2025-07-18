enroot create -n mlr3torch-jss sebffischer+mlr3torch-jss+latest.sqsh 
enroot start mlr3torch-jss
conda deactivate
cd ~/mlr3torch/paper
Rscript benchmark/benchmark.R