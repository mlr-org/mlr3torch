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
    build = function(input, task) {
      # TODO: Dlo checks
      param_vals = self$param_set$get_values(tag = "train")
      private$.build(input, param_vals, task)
    },
    #' Provides the repreesntation for the TorchOp.
    repr = function() {
      sprinf("<%s: %s>", self$.operator, self$param_set$get_values(tag = "train"))
    }
  ),
  active = list(
    .operator = function() {
      formals(self$initialize)[["id"]]
    }
  ),
  private = list(
    .train = function(inputs) {
      if (!is.null(self$state)) {
        # architecture is already built
        return(list(task = inputs[["task"]], architecture = NULL))
      }
      task = inputs[["task"]]
      architecture = inputs[["architecture"]]
      architecture$add(self$build)
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
    },
    .output_dim = function(input_dim) {
      stop("ABC")
    }
  )
)
