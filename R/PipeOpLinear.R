#' @export
PipeOpLinear = R6::R6Class("PipeOpLinear",
  inherit = mlr3pipelines::PipeOp,
  public = list(
    initialize = function(id = "linear", param_vals = list()) {
      param_set = ps(
        units = p_int(1L, Inf, tags = "train"),
        bias = p_lgl(default = TRUE, tags = "train")
      )
      input = data.table(
        name = c("task", "architecture"),
        train = c("Task", "Architecture"),
        predict = c("Task", "*")
      )
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
        output = output,
        packages = "torch",
      )
    }
  ),
  private = list(
    .train = function(inputs) {
      if (!is.null(self$state)) {
        # architecture is already built
        return(list(task = inputs[["task"]], NULL))
      }
      task = inputs[["task"]]
      architecture = inputs[["architecture"]]
      architecture$append(
        list(
          bob = private$.bob,
          param_vals = self$param_set$values
        )
      )
      self$state = "trained"
      output = list(task = inputs[["task"]], architecture = architecture)
      return(output)
    },
    .predict = function(inputs) {
      task = inputs[["task"]]
      architecture = inputs[["architecture"]]
      output = list(task = task, architecture = architecture)
      return(output)
    },
    # Bob the builder
    .bob = function(x, network, param_vals) {
      layer = torch::nn_linear(
        in_features = dim(x[[2L]]),
        out_features = param_set$values$units,
        bias = param_set$values$bias
      )
      return(layer)
    }
  )
)

if (FALSE) {
  task = tsk("iris")
  pipeop_input = PipeOpInput$new()
  pipeop_linear = PipeOpLinear$new()
  graph = pipeop_input %>>%
    pipeop_linear

  graph$train(task)

}
