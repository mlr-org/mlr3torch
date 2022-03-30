#' @export
TorchOpSelfAttention = R6Class("TorchOpSelfAttention",
  inherit = TorchOp,
  public = list(
    initialize = function(id = "selfattention", param_vals = list()) {
      param_set = ps(
        d_token = p_int(default = NO_DEF, lower = 1L, tags = "train"),
        n_heads = p_int(default = 1L, lower = 1L, tags = "train"),
        dropout = p_dbl(default = 0.5, lower = 0, upper = 1, tags = "train"),
        bias = p_lgl(default = TRUE, tags = "train"),
        # TODO: correct default?
        initialization = p_fct(default = "kaiming", levels = c("kaiming", "xavier"))
      )
      super$initialize(
        id = id,
        param_set = param_set,
        param_vals = param_vals
      )
    },
    names_in = c("batch", "feature"),
    names_out = function(names_in) {
      c("batch", "feature", "token")
    }
  ),
  private = list(
    .operator = "selfattention",
    .build = function(inputs, param_vals, task, y) {
      layer = invoke(nn_self_attention, .args = param_vals)
      return(layer)
    }
  )
)

#' @include mlr_torchops.R
mlr_torchops$add("selfattention", TorchOpSelfAttention)
