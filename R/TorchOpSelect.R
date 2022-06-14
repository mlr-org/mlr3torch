#' @export
TorchOpSelect = R6Class("TorchOpSelect",
  inherit = TorchOp,
  public = list(
    initialize = function(id = "select", param_vals = list(), .items) {
      private$.items = .items
      param_set = ps()
      super$initialize(
        id = id,
        param_set = param_set,
        param_vals = param_vals
      )
    }
  ),
  private = list(
    .build = function(inputs, param_vals, task) {
      input = inputs$input
      assert_subset(private$.items, names(input))
      layer = nn_select(items = private$.items)
      return(layer)
    },
    .items = NULL
  )
)

nn_select = nn_module("nn_select",
  initialize = function(items) {
    self$items = items
  },
  forward = function(input) {
    if (length(self$items == 1L)) {
      input[[self$items]]
    } else {
      input[self$items]
    }
  }
)

#' @include mlr_torchops.R
mlr_torchops$add("select", TorchOpSelect)
