
library(batchtools)
library(data.table)

reg = loadRegistry("/mnt/data/mlr3torch/paper/benchmark/registry", writeable = TRUE)
tbl = unwrap(getJobTable())
ids = tbl[device == "cpu" & optimizer == "sgd" & repl == 1L & algorithm != "pytorch" & !jit, ]$job.id
for (id in sample(ids)) {
  submitJobs(id)
  Sys.sleep(0.1)
}
