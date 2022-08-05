#' @title Initialization of ModelConfig
#'
#' @description
#' Outputs an object of class `"ModelConfig"` that is used internally to build Torch Learners using
#' `TorchOp`'s.
#'
#' @template param_id
#' @template param_param_vals
#' @export
TorchOpInput = R6Class("TorchOpInput",
  inherit = TorchOp,
  public = list(
    #' @description Initializes an instance of this [R6][R6::R6Class] class.
    initialize = function(id = "input", param_vals = list()) {
      input = data.table(name = "input", train = "Task", predict = "Task")
      output = data.table(name = "output", train = "ModelConfig", predict = "Task")
      param_set = ps()

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
      instance = get_batch(inputs$input, batch_size = 1L, device = "cpu")
      y = instance$y
      x = instance$x

      self$state = list()
      model_args = structure(
        class = "ModelConfig",
        list(
          network = nn_graph(),
          id = "__initial__",
          task = inputs$input,
          channel = "output",
          output = x,
          y = y
        )
      )
      list(output = model_args)
    },
    .predict = function(inputs) {
      inputs
    }
  )
)


#' @include mlr_torchops.R
mlr_torchops$add("input", value = TorchOpInput)
