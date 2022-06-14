#' @title Tabular ResNet Block
#' @description
#' This block is mostly taken from The RTDL paper.
#'
#' @export
#'
#' @references `r format_bib("gorishniy2021revisiting")`
TorchOpTabResNetBlock = R6Class("TorchOpBlockTabResNetBlock",
  inherit = TorchOp,
  public = list(
    initialize = function(id = "tabular_resnet", param_vals = list()) {
      param_set = make_paramset_tab_resnet_block()
      super$initialize(id = id, param_set = param_set, param_vals = param_vals)
    }
  ),
  private = list(
    .build = function(inputs, param_vals, task) {
      n_blocks = param_vals$n_blocks
      param_vals["n_blocks"] = NULL
      ii = startsWith(names(param_vals), "batch_norm")
      batch_norm_args = param_vals[ii]
      param_vals[ii] = NULL

      input_layer = top("linear", out_features = param_vals$d_main)$build(inputs, task)$layer

      blocks = replicate(
        n = n_blocks,
        invoke(nn_tab_resnet_block, .args = param_vals, batch_norm_args = batch_norm_args)
        )
      layer = invoke(nn_sequential, .args = c(input_layer, blocks))

      return(layer)
    }
  )
)

check_activation_args = function(x) {
  check_list(x, names = "unique", null.ok = TRUE, any.missing = FALSE)
}


#' @include mlr_torchops.R
mlr_torchops$add("tab_resnet_block", TorchOpTabResNetBlock)


nn_tab_resnet_block = nn_module("tab_resnet_block",
  initialize = function(
    d_main,
    d_hidden,
    dropout_first,
    dropout_second,
    skip_connection = TRUE,
    batch_norm_args,
    activation,
    activation_args = list()
    ) {
    self$normalization = invoke(nn_batch_norm1d, num_features = d_main, .args = batch_norm_args)
    self$activation = invoke(get_activation(activation), .args = activation_args)
    self$linear_first = nn_linear(d_main, d_hidden, TRUE)
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
  },
  reset_parameters = function() {
    walk(self$modules[-1], # 1 is the module itself
      function(mod) {
        if (!is.null(mod$reset_parameters)) {
          mod$reset_parameters()
        }
      }
    )
  },
  reset_running_stats = function() {
    walk(self$modules[-1], # 1 is the module itself
      function(mod) {
        if (!is.null(mod$reset_running_stats)) {
          mod$reset_running_stats()
        }
      }
    )
  }
)

make_paramset_tab_resnet_block = function() {
  psc = ParamSetCollection$new(list())
  ps1 = ps(
    n_blocks = p_int(lower = 1, tags = c("required", "train")),
    d_main = p_int(tags = c("train", "required")),
    d_hidden = p_int(tags = c("train", "required")),
    dropout_first = p_dbl(lower = 0, upper = 1, tags = c("train", "required")),
    dropout_second = p_dbl(lower = 0, upper = 1, tags = c("train", "required")),
    skip_connection = p_lgl(default = TRUE, tags = c("train")),
    activation = p_fct(default = "relu", levels = torch_reflections$activation,
      tags = c("required", "train")),
    activation_args = p_uty(default = list(), tags = "train")
  )
  ps2 = ps(
    eps = p_dbl(default = 1e-05, lower = 0, tags = "train"),
    momentum = p_dbl(default = 0.1, lower = 0, tags = "train"),
    affine = p_lgl(default = TRUE, tags = "train"),
    track_running_stats = p_lgl(default = TRUE, tags = "train")
  )
  ps2$set_id = "batch_norm"
  # ps2$values = list( momentum = 0.1, )

  psc$add(ps1)$add(ps2)

  psc$values = list(
    skip_connection = TRUE,
    activation = "relu",
    activation_args = list()
  )

  psc
}
