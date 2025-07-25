
library(batchtools)
library(data.table)

reg = loadRegistry("~/mlr3torch/paper/benchmark/registry", writeable = TRUE)
tbl = unwrap(getJobTable())
ids = tbl[device == "cpu", ]$job.id
for (id in sample(ids)) {
  submitJobs(id)
  Sys.sleep(0.1)
}