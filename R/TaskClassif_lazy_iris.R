#' @title Iris Classification Task
#'
#' @name mlr_tasks_lazy_iris
#'
#' @format [R6::R6Class] inheriting from [mlr3::TaskClassif].
#' @include zzz.R
#'
#' @description
#' A classification task for the popular [datasets::iris] data set.
#' Just like the iris task, but the features are represented as one lazy tensor column.
#'
#' @section Construction:
#' ```
#' tsk("lazy_iris")
#' ```
#' @source
#' \url{https://en.wikipedia.org/wiki/Iris_flower_data_set}
#'
#' @section Properties:
#' `r rd_info_task_torch("lazy_iris", missings = FALSE)`
#'
#' @references
#' `r format_bib("anderson_1936")`
#' @examplesIf torch::torch_is_installed()
#' task = tsk("lazy_iris")
#' task
#' df = task$data()
#' materialize(df$x[1:6], rbind = TRUE)
NULL

load_task_lazy_iris = function(id = "lazy_iris") {
  d = load_dataset("iris", "datasets")
  target = d[, 5]
  features = as.matrix(d[, -5])

  d = data.table(
    Species = target,
    x = as_lazy_tensor(features)
  )

  b = as_data_backend(d)
  task = TaskClassif$new(id, b, target = "Species", label = "Iris Flowers")
  b$hash = task$man = "mlr3torch::mlr_tasks_lazy_iris"
  task
}

#' @include zzz.R
mlr3torch_tasks[["lazy_iris"]] = load_task_lazy_iris
