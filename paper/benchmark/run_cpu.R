
library(batchtools)
library(data.table)

reg = loadRegistry("/mnt/data/mlr3torch/paper/benchmark/registry", writeable = TRUE)
tbl = unwrap(getJobTable())
tbl = tbl[device == "cpu", ]
ids = tbl$job.id
for (id in sample(ids)) {
  submitJobs(id)
  Sys.sleep(0.1)
}
