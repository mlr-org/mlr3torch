#' @export
TorchOpInput = R6Class("TorchOpInput",
  inherit = TorchOp,
  public = list(
    initialize = function(id = "input", param_vals = list()) {
      param_set = ps()
      input = data.table(name = "task", train = "Task", predict = "Task")
      output = data.table(
        name = c("task", "architecture"),
        train = c("Task", "Architecture"),
        predict = c("Task", "*")
      )

      super$initialize(
        id = id,
        param_set = param_set,
        param_vals = param_vals,
        input = input,
        output = output
      )
    }
  ),
  private = list(
    .train = function(inputs) {
      self$state = list()
      list(task = inputs[["task"]], architecture = Architecture$new())
    },
    .predict = function(inputs) {
      list(inputs[["task"]], architecture = NULL)
    }
  )
)

#' @include mlr_torchops.R
mlr_torchops$add("input", value = TorchOpInput)
