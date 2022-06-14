#' @title Tabular Attention Mechanism from RTDL paper
#' @description
#' This is a R reimplementan of the tabular attention layer as implemented in the paper
#' "revisiting tabular deep learning".
#' @export
TorchOpRtdlAttention = R6Class("TorchOpAttention",
  inherit = TorchOp,
  public = list(
    #' @description Initializes an instance of this [R6][R6::R6Class] class.
    #' @param id (`character(1)`)\cr
    #'   The id for of the object.
    #' @param param_vals (named `list()`)\cr
    #'   The initial parameters for the object.
    initialize = function(id = "rdtl_attention", param_vals = list()) {
      param_set = ps(
        n_heads = p_int(default = 1L, lower = 1L, tags = "train"),
        dropout = p_dbl(default = 0.5, lower = 0, upper = 1, tags = "train"),
        bias = p_lgl(default = TRUE, tags = "train"),
        # TODO: correct default?
        initialization = p_fct(default = "kaiming", levels = c("kaiming", "xavier"))
      )
      input = data.table(
        name = c("query", "key"),
        train = c("ModelArgs", "ModelArgs"),
        predict = c("*", "*")
      )
      super$initialize(
        id = id,
        param_vals = param_vals,
        input = input,
        param_set = param_set
      )
    }
  ),
  private = list(
    .build = function(inputs, param_vals, task) {
      query = inputs$query
      key = inputs$key
      # TODO: What exactly are the assumptions here?
      assert_true(length(query$shape) == 3L)
      assert_true(length(key$shape) == 3L)
      assert_true(all.equal(query$shape[2:3], key$shape[2:3]))
      d_token = query$shape[[3L]]
      invoke(nn_rtdl_attention, .args = param_vals, d_token = d_token)
    }
  )
)

#' @include mlr_torchops.R
mlr_torchops$add("rtdl_attention", TorchOpRtdlAttention)
