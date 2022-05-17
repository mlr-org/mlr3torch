TorchOpSelect = R6Class("TorchOpSelect",
  inherit = TorchOp,
  public = list(
    #' @description Initializes an instance of this [R6][R6::R6Class] class.
    #' @param id (`character(1)`)\cr
    #'   The id for of the object.
    #' @param param_vals (named `list()`)\cr
    #'   The initial parameters for the object.
    initialize = function(id = "select", param_vals = list(), .types) {
      private$.types = .types
      assert_character(.types, min.len = 1L)
      super$initialize(
        id = id,
        param_vals = param_vals,
        param_set = ps()
      )
    }
  ),
  private = list(
    .build = function(inputs, param_vals, task, y) {
      assert_list(inputs)
      nn_module(
        initialize = function(types) {
          self$types = types
        },
        forward = function(inputs) {
          if (length(self$types) > 1L) {
            return(inputs[self$types])
          }
          inputs[[self$types]]
        }
      )$new(private$.types)
    },
    .types = NULL
  )
)

mlr_torchops$add("select", TorchOpSelect)
