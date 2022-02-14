#'  @title Linear TorchOp
#' @include TorchOpLinear.R
#' @export
TorchOpLinear = R6::R6Class("TorchOpLinear",
  inherit = TorchOp,
  public = list(
    initialize = function(id = "linear", param_vals = list()) {
      param_set = ps(
        out_features = p_int(1L, Inf, tags = c("train", "required")),
        bias = p_lgl(default = TRUE, tags = "train")
      )
      super$initialize(
        id = id,
        param_set = param_set,
        param_vals = param_vals
      )
    }
  ),
  private = list(
    .operator = "linear",
    .build = function(input, param_vals, task) {
      # TODO: Do this prettier
      assert_true("y" %in% names(input))
      assert_true(sum(startsWith(names(input), "x")) == 1)
      in_features = dim(input[startsWith(names(input), "x")][[1L]])[[2L]]
      assert_true(length(input) == 2)
      layer = invoke(nn_linear, in_features = in_features, .args = param_vals)
      return(layer)
    }
  )
)

#' @include mlr_torchops.R
mlr_torchops$add("linear", value = TorchOpLinear)
# .__bobs__.[["linear"]] = TorchOpLinear$private_methods$.build
