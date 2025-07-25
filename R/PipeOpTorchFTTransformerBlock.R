#' @title Single Transformer Block for FT-Transformer
#' @description
#' A transformer block consisting of a multi-head self-attention mechanism followed by a feed-forward
#' network.
#'
#' This is used in [`LearnerTorchFTTransformer`].
#'
#' @param d_token (`integer(1)`)\cr
#'   The dimension of the embedding.
#' @param attention_n_heads (`integer(1)`)\cr
#'   Number of attention heads.
#' @param attention_dropout (`numeric(1)`)\cr
#'   Dropout probability in the attention mechanism.
#' @param attention_initialization (`character(1)`)\cr
#'   Initialization method for attention weights. Either "kaiming" or "xavier".
#' @param ffn_d_hidden (`integer(1)`)\cr
#'   Hidden dimension of the feed-forward network. Multiplied by 2 if using ReGLU or GeGLU activation.
#' @param ffn_d_hidden_multiplier (`numeric(1)`)\cr
#'   Alternative way to specify the hidden dimension of the feed-forward network as `d_token * d_hidden_multiplier`. Also multiplied by 2 if using RegLU or GeGLU activation.
#' @param ffn_dropout (`numeric(1)`)\cr
#'   Dropout probability in the feed-forward network.
#' @param ffn_activation (`nn_module`)\cr
#'   Activation function for the feed-forward network. Default value is `nn_reglu`.
#' @param residual_dropout (`numeric(1)`)\cr
#'   Dropout probability for residual connections.
#' @param prenormalization (`logical(1)`)\cr
#'   Whether to apply normalization before attention and FFN (`TRUE`) or after (`TRUE`).
#' @param is_first_layer (`logical(1)`)\cr
#'   Whether this is the first layer in the transformer stack. Default value is `FALSE`.
#' @param attention_normalization (`nn_module`)\cr
#'   Normalization module to use for attention. Default value is `nn_layer_norm`.
#' @param ffn_normalization (`nn_module`)\cr
#'   Normalization module to use for the feed-forward network. Default value is `nn_layer_norm`.
#' @param query_idx (`integer()` or `NULL`)\cr
#'   Indices of the tensor to apply attention to. Should not be set manually.
#'   If NULL, then attention is applied to the entire tensor.
#'   In the last block in a stack of transformers, this is set to `-1`
#'   so that attention is applied only to the embedding of the CLS token.
#' @param attention_bias (`logical(1)`)\cr
#'   Whether attention has a bias. Default is `TRUE`
#' @param ffn_bias_first (`logical(1)`)\cr
#'   Whether the first layer in the FFN has a bias. Default is `TRUE`
#' @param ffn_bias_second (`logical(1)`)\cr
#'   Whether the second layer in the FFN has a bias. Default is `TRUE`
#'
#' @references
#' `r format_bib("devlin2018bert")`
#' `r format_bib("gorishniy2021revisiting")`
#'
#' @export
nn_ft_transformer_block = nn_module(
  "nn_ft_transformer_block",
  initialize = function(d_token,
    attention_n_heads,
    attention_dropout,
    attention_initialization,
    ffn_d_hidden = NULL,
    ffn_d_hidden_multiplier = NULL,
    ffn_dropout,
    ffn_activation,
    residual_dropout,
    prenormalization,
    is_first_layer,
    attention_normalization,
    ffn_normalization,
    query_idx = NULL,
    attention_bias,
    ffn_bias_first,
    ffn_bias_second
  ) {

    if ((!is_first_layer) || (!prenormalization)) {
      self$attention_normalization = attention_normalization(d_token)
    }

    self$ffn_normalization = ffn_normalization(d_token)

    self$prenormalization = prenormalization

    self$attention = nn_multihead_attention(
      embed_dim = d_token,
      num_heads = attention_n_heads,
      dropout = attention_dropout,
      bias = attention_bias,
      batch_first = TRUE
    )

    if (is.null(ffn_d_hidden)) {
      assert_numeric(ffn_d_hidden_multiplier, lower = 0,
        .var.name = "Exactly one of ffn_d_hidden and ffn_d_hidden_multiplier is set")
      d_hidden = round(ffn_d_hidden_multiplier * d_token)
    } else {
      assert_int(ffn_d_hidden, lower = 1L)
      assert_true(is.null(ffn_d_hidden_multiplier),
        .var.name = "Exactly one of ffn_d_hidden and ffn_d_hidden_multiplier is set")
      d_hidden = ffn_d_hidden
    }

    self$ffn = nn_ft_ffn(
      d_token = d_token,
      d_hidden = d_hidden,
      bias_first = ffn_bias_first,
      bias_second = ffn_bias_second,
      dropout = ffn_dropout,
      activation = ffn_activation
    )

    self$attention_residual_dropout = nn_dropout(residual_dropout)
    self$ffn_residual_dropout = nn_dropout(residual_dropout)

    self$query_idx = query_idx
  },
  start_residual_ = function(stage, x) {
    x_residual = x
    if (self$prenormalization) {
      norm_key = paste0(stage, "_normalization")
      if (norm_key %in% names(self)) {
        x_residual = self[[norm_key]](x_residual)
      }
    }
    return(x_residual)
  },
  end_residual_ = function(stage, x, x_residual) {
    x_residual = self[[paste0(stage, "_residual_dropout")]](x_residual)
    x = x + x_residual
    if (!self$prenormalization) {
      x = layer[[paste0(stage, "_normalization")]](x)
    }
    return(x)
  },
  forward = function(x) {
    x_residual = self$start_residual_("attention", x)

    x_residual_arg = if (is.null(self$query_idx)) x_residual else x_residual[, self$query_idx, drop = FALSE]
    x_residual = self$attention(
      query = x_residual_arg,
      key = x_residual,
      value = x_residual,
      need_weights = FALSE
    )[[1L]]
    x = if (!is.null(self$query_idx)) x[, self$query_idx, drop = FALSE] else x
    x = self$end_residual_("attention", x, x_residual)

    x_residual = self$start_residual_("ffn", x)
    x_residual = self$ffn(x_residual)
    x = self$end_residual_("ffn", x, x_residual)

    return(x)
  }
)

#' @title Single Transformer Block for the FT-Transformer
#' @inherit nn_ft_transformer_block description
#' @section nn_module:
#' Calls [`nn_ft_transformer_block()`] when trained.
#' @templateVar id nn_ft_transformer_block
#' @template pipeop_torch
#' @template pipeop_torch_example
#' @export
PipeOpTorchFTTransformerBlock = R6::R6Class("PipeOpTorchFTTransformerBlock",
  inherit = PipeOpTorch,
  lock_objects = FALSE,
  public = list(
    #' @description Create a new instance of this [R6][R6::R6Class] class.
    #' @template params_pipelines
    initialize = function(id = "nn_ft_transformer_block", param_vals = list()) {
      param_set = ps(
        attention_n_heads = p_int(lower = 1L, init = 8L, tags = "train"),
        attention_dropout = p_dbl(lower = 0, upper = 1, init = 0.2, tags = "train"),
        attention_initialization = p_fct(levels = c("kaiming", "xavier"), init = "kaiming", tags = "train"),
        attention_normalization = p_uty(init = nn_layer_norm, custom_check = check_nn_module_generator, tags = "train"),
        ffn_d_hidden = p_int(lower = 1, tags = "train"),
        ffn_d_hidden_multiplier = p_dbl(lower = 0, tags = "train"),
        ffn_dropout = p_dbl(lower = 0, upper = 1, init = 0.1, tags = "train"),
        ffn_activation = p_uty(init = nn_reglu, custom_check = check_nn_module_generator, tags = "train"),
        ffn_normalization = p_uty(init = nn_layer_norm, custom_check = check_nn_module_generator, tags = "train"),
        residual_dropout = p_dbl(lower = 0, upper = 1, init = 0.0, tags = "train"),
        prenormalization = p_lgl(init = TRUE, tags = "train"),
        is_first_layer = p_lgl(init = FALSE, tags = "train"),
        query_idx = p_uty(init = NULL, custom_check = function(input) check_integerish(input, null.ok = TRUE), tags = "train"),
        attention_bias = p_lgl(init = TRUE, tags = "train"),
        ffn_bias_first = p_lgl(init = TRUE, tags = "train"),
        ffn_bias_second = p_lgl(init = TRUE, tags = "train")
      )

      super$initialize(
        id = id,
        module_generator = nn_ft_transformer_block,
        param_vals = param_vals,
        param_set = param_set
      )
    }
  ),
  private = list(
    .shapes_out = function(shapes_in, param_vals, task) {
      if (is.null(param_vals$query_idx)) {
        return(shapes_in[1])
      }

      shapes_out = shapes_in$input
      # to save computation, apply the last transformer block to only the CLS token
      shapes_out[[2L]] = 1
      return(list(shapes_out))
    },
    .shape_dependent_params = function(shapes_in, param_vals, task) {
      param_vals$d_token = shapes_in$input[3]
      return(param_vals)
    }
  )
)

#' @include aaa.R
register_po("nn_ft_transformer_block", PipeOpTorchFTTransformerBlock)

nn_ft_ffn = nn_module(
  "nn_ft_ffn",
  initialize = function(d_token, d_hidden, bias_first, bias_second, dropout, activation) {
    # ReGLU, GeGLU activations change the size of their input
    activation_coef = if (class(activation)[1] %in% c("nn_reglu", "nn_geglu")) 2 else 1
    self$linear_first = nn_linear(d_token, d_hidden * activation_coef, bias_first)
    self$activation = activation()
    self$dropout = nn_dropout(dropout)
    self$linear_second = nn_linear(d_hidden, d_token, bias_second)
  },
  forward = function(x) {
    x = self$linear_first(x)
    x = self$activation(x)
    x = self$dropout(x)
    x = self$linear_second(x)
    return(x)
  }
)
