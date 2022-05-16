#' Parameters:
#'  - simplify: whether to simplfy the output of the dataloader
#' @export
TorchOpInput = R6Class("TorchOpInput",
  inherit = TorchOp,
  public = list(
    #' @description Initializes an instance of this [R6][R6::R6Class] class.
    #' @param id (`character(1)`)\cr
    #'   The id for of the object.
    #' @parm param_vals (named `list()`)\cr
    #'   The initial parameters for the object.
    initialize = function(id = "input", param_vals = list()) {
      input = data.table(name = "task", train = "Task", predict = "Task")
      output = data.table(name = "output", train = "ModelArgs", predict = "Task")
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
      self$state = list()
      model_args = structure(class = "ModelArgs",
        list(
          architecture = Architecture$new(),
          task = inputs$task,
          id = NULL
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
