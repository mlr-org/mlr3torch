#' @export
TorchOpSelect = R6Class("TorchOpSelect",
  inherit = TorchOp,
  public = list(
    initialize = function(id = "select", param_vals = list()) {
      param_set = ps(
        items = p_uty(tags = c("required", "train"), custom_check = check_select)
      )
      super$initialize(
        id = id,
        param_set = param_set,
        param_vals = param_vals
      )
    }
  ),
  private = list(
    .build = function(inputs, task) {
      param_vals = self$param_set$get_values(tag = "train")
      input = inputs$input
      items = param_vals$items
      assert_list(input)
      assert_subset(items, names(input))
      invoke(nn_select, .args = param_vals)
    }
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


check_select = function(x) {
  if (is.null(x)) {
    return(TRUE)
  } else if (test_subset(x, c("img", "num", "cat"))) {
    return(TRUE)
  }
  "Must be subset of c(\"img\", \"num\", \"cat\")"
}
