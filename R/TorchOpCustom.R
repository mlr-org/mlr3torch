#' @title Custom Torch Operator
#' @description This TorchOp allows to create a custom torch operator.
#' @details
#' build = f(tensor, ...) --> nn_module()
#' args: the ... in the call above
#'
#' @include TorchOp.R
#' @export
TorchOpCustom = R6Class("TorchOpCustom",
  inherit = TorchOp,
  public = list(
    #' @description Initializes an instance of this [R6][R6::R6Class] class.
    #' @param id (`character(1)`)\cr
    #'   The id for of the object.
    #' @parm param_vals (named `list()`)\cr
    #'   The initial parameters for the object.
    initialize = function(id = "custom", param_vals = list()) {
      param_set = ps(
        fn = p_uty(tags = c("train", "required"), custom_check = check_fn),
        args = p_uty(tags = "train")
      )
      # TODO: add dependency that args specifies only values that are in fn
      super$initialize(
        id = id,
        param_set = param_set,
        param_vals = param_vals
      )
    }
  ),
  private = list(
    .operator = "custom",
    .build = function(inputs, task, param_vals, y) {
      pars = self$param_set$get_values(tags = "train")
      fn = pars[["fn"]]
      args = pars[["args"]]
      invoke(fn, inputs, .args = args)
    }
  )
)


check_fn = function(x) {
  assert("input" %in% formalArgs(x))
}
