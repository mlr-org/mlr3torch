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
          name = "input",
          train = "*",
          predict = "*"
        )
      }
      if (is.null(output)) {
        output = data.table(
          name = "output",
          train = "*",
          predict = "*"
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
    build = function(input, task, y) {
      # TODO: Dlo checks
      param_vals = self$param_set$get_values(tag = "train")
      private$.build(
        input = input,
        param_vals = param_vals,
        task = task,
        y = y
      )
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
      # TODO: input checks: either task or list(task, architecture)
      if (!is.null(self$state)) { # this means the architecture is already built
        return(inputs)
      }
      if (test_r6(inputs[[1L]], "Task")) { # this means this torchop is the first in the architecture
        # and we have to build the architecture
        task = inputs[[1L]]
        architecture = Architecture$new()
      } else {
        task = inputs[["input"]][["task"]]
        architecture = inputs[["input"]][["architecture"]]
      }
      architecture$add_node(self)
      self$state = list()
      output = list(list(task = task, architecture = architecture))
      return(output)
    },
    .predict = function(inputs) {
      inputs = inputs[[1L]]
      task = inputs[["task"]]
      architecture = inputs[["architecture"]]
      output = list(task = task, architecture = architecture)
      return(output)
    },
    .build = function(input, param_vals, task, y) {
      stop("ABC")
    },
    .output_dim = function(input_dim) {
      stop("ABC")
    }
  )
)
