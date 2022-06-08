#' @title Tabular ResNet Block
#' @description
#' This block is mostly taken from The RTDL paper.
#'
#' @export
#'
#' @references `r format_bib("gorishniy2021revisiting")`
TorchOpTabResNet = R6Class("TorchOpTabResNet",
  inherit = TorchOp,
  public = list(
    initialize = function(id = "tabular_resnet", param_vals = list()) {
      param_set = ps(
        n_blocks = p_int(lower = 1, tags = c("required", "train")),
        d_main = p_int(tags = c("train", "required")),
        d_hidden = p_int(tags = c("train", "required")),
        dropout_first = p_dbl(lower = 0, upper = 1, tags = c("train", "required")),
        dropout_second = p_dbl(lower = 0, upper = 1, tags = c("train", "required")),
        # TODO: Maybe add LayerNorm when I properly understand it
        normalization = p_fct(levels = c("batch_norm"), tags = c("train", "required")),
        activation = p_fct(levels = torch_reflections$activation, tags = c("train", "required")),
        skip_connection = p_lgl(tags = c("train", "required"))
      )
      param_set$values = list(
        normalization = "batch_norm"
      )

      super$initialize(id = id, param_set = param_set, param_vals = param_vals)
    }
  ),
  private = list(
    .build = function(inputs, param_vals, task, y) {
      n_blocks = param_vals$n_blocks
      param_vals$n_blocks = NULL

      blocks = replicate(n_blocks, invoke(nn_block_resnet, .args = param_vals))
      invoke(nn_sequential, .args = blocks)
    }
  )
)


#' @include mlr_torchops.R
mlr_torchops$add("tab_resnet", TorchOpTabResNet)


nn_block_resnet = nn_module("block_resnet",
  initialize = function(task, n_blocks, d_main, d_hidden, dropout_first, dropout_second,
    normalization, activation, skip_connection) {
    self$normalization = switch(normalization,
      batch_norm = nn_batch_norm1d(d_main),
      stopf("Not implemented yet.")
    )
    self$linear_first = nn_linear(d_main, d_hidden, TRUE)
    self$activation = get_activation(activation)()
    self$dropout_first = nn_dropout(dropout_first)
    self$linear_second = nn_linear(d_hidden, d_main, TRUE)
    self$dropout_second = nn_dropout(dropout_second)
    self$skip_connection = skip_connection

  },
  forward = function(input) {
    x = self$normalization(input)
    x = self$linear_first(x)
    x = self$activation(x)
    x = self$dropout_first(x)
    x = self$linear_second(x)
    x = self$dropout_second(x)
    if (self$skip_connection) {
      x = input + x
    }
    return(x)
  }
)
