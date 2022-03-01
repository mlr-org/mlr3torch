#' @title Abstract Base Class for Torch Operators
#' @description All TorchOps inherit from this class.
#' @export
TorchOp = R6Class("TorchOp",
  inherit = mlr3pipelines::PipeOp,
  public = list(
    initialize = function(id, param_set, param_vals, input = NULL, output = NULL,
      packages = NULL) {
      if (is.null(input)) {
        input = data.table(
          name = c("task", "architecture"),
          train = c("Task", "Architecture"),
          predict = c("Task", "*")
        )
      }
      if (is.null(output)) {
        output = data.table(
          name = c("task", "architecture"),
          train = c("Task", "Architecture"),
          predict = c("Task", "*")
        )
      }
      if (is.null(packages)) {
        packages = "torch"
      }
      super$initialize(
        id = id,
        param_set = param_set,
        param_vals = param_vals,
        input = input,
        output = output,
        packages = packages
      )
    },
    #' Builds the torch layer
    #' @param input (torch_tensor) The torch tensor that is the input to this layer.
    #' @param param_vals (list()) parameter values passed to the build function.
    #' @param task (mlr3::Task) The task on which the architecture is trained.
    build = function(input, param_vals, task) {
      # TODO: Do checks

      private$.build(input, param_vals, task)
    },
    repr = function(param_vals) {
      sprinf("<%s: %s>", self$.operator, param_vals)
    }
  ),
  private = list(
    .operator = "abstract",
    .train = function(inputs) {
      if (!is.null(self$state)) {
        # architecture is already built
        return(list(task = inputs[["task"]], architecture = NULL))
      }
      task = inputs[["task"]]
      architecture = inputs[["architecture"]]
      architecture$add(self$build, self$param_set$get_values(tag = "train"))
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
    .build = function(input, param_vals, task) {
      stop("ABC")
    }
  )
)
