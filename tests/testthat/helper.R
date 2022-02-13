library(checkmate)
library(mlr3)
library(mlr3misc)
# Load test scaffolding without helper_debugging.R temporarily
# See https://github.com/mlr-org/mlr3torch/issues/8
lapply(list.files(system.file("testthat", package = "mlr3"), pattern = "^helper.*\\.[rR]$", full.names = TRUE)[-2], source)

make_mtcars_task = function() {
  data = mtcars
  data[["..row_id"]] = seq_len(nrow(data))
  data = as.data.table(data)
  backend = DataBackendTorchDataTable$new(data = data, primary_key = "..row_id")
  task = TaskRegr$new(
    id = "mtcars",
    backend = backend,
    target = "mpg"
  )
  return(task)
}
