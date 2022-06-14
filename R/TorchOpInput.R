#' @title Initialization of ModelArgs
#'
#' @description
#' Outputs an object of class `"ModelArgs"` that is used internally to build Torch Learners using
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
      input = data.table(name = "task", train = "Task", predict = "Task")
      output = data.table(name = "output", train = "ModelArgs", predict = "Task")
      param_set = ps(
        select = p_uty(tags = "train", custom_check = check_select)
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
      instance = get_batch(inputs$task, batch_size = 1L, device = "cpu")
      y = instance$y
      x = instance$x

      self$state = list()
      model_args = structure(
        class = "ModelArgs",
        list(
          network = nn_graph$new(),
          task = inputs$task,
          id = "__initial__",
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

check_select = function(x) {
  if (is.null(x)) {
    return(TRUE)
  } else if (test_subset(x, c("img", "num", "cat"))) {
    return(TRUE)
  }
  "Must be subset of c('img', 'num', 'cat')"
}

#' @include mlr_torchops.R
mlr_torchops$add("input", value = TorchOpInput)
