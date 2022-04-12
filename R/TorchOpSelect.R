TorchOpSelect = R6Class("TorchOpSelect",
  inherit = TorchOp,
  public = list(
    types = NULL,
    initialize = function(id, param_vals, types) {
      self$types = types
      assert_char(types, min.len = 1L)
    }
  ),
  private = list(
    .build = function(input, param_vals, task, y) {
      assert_list(input)
      types = self$types
      nn_module(
        initialize = function(types) {
          self$types = types
        },
        forward = function(inputs) {
          if (length(self$types) > 1L) {
            return(inputs[self$types])
          }
          input[[types]]
        }
      )$new()
    }
  )
)

mlr_torchops$add("select", TorchOpSelect)
