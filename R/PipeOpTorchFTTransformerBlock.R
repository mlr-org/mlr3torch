#' @title Single Transformer Block for FT-Transformer
#' @description
#' A transformer block, consisting of a multi-head self-attention mechanism followed by a feed-forward
#' network.
#'
#' This is used in the FT-Transformer.
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
#'   Hidden dimension of the feed-forward network.
#' @param ffn_dropout (`numeric(1)`)\cr
#'   Dropout probability in the feed-forward network.
#' @param ffn_activation (`nn_module`)\cr
#'   Activation function for the feed-forward network. Default value is `nn_reglu`.
#' @param residual_dropout (`numeric(1)`)\cr
#'   Dropout probability for residual connections.
#' @param prenormalization (`logical(1)`)\cr
#'   Whether to apply normalization before attention and FFN (TRUE) or after (FALSE).
#' @param is_first_layer (`logical(1)`)\cr
#'   Whether this is the first layer in the transformer stack. Default value is FALSE.
#' @param attention_normalization (`function`)\cr
#'   Normalization function to use for attention. Default value is `nn_layer_norm`.
#' @param ffn_normalization (`function`)\cr
#'   Normalization function to use for the feed-forward network. Default value is `nn_layer_norm`.
#' @param query_idx (`integer()` or `NULL`)\cr
#'   Indices to select for the query.
#' @param attention_bias (`logical(1)`)\cr
#'   Whether attention is biased. Default is TRUE.
#' @param ffn_bias_first (`logical(1)`)\cr
#'   Whether the first layer in the FFN has a bias. Default is TRUE.
#' @param ffn_bias_second (`logical(1)`)\cr
#'   Whether the second layer in the FFN has a bias. Default is TRUE.
#'
#' @references
#' `r format_bib("devlin2018bert")`
#'
#' @export
nn_ft_transformer_block = nn_module(
  "nn_ft_transformer_block",
  initialize = function(d_token,
    attention_n_heads,
    attention_dropout,
    attention_initialization,
    ffn_d_hidden,
    ffn_dropout,
    ffn_activation,
    residual_dropout,
    prenormalization,
    is_first_layer,
    attention_normalization,
    ffn_normalization,
    query_idx,
    attention_bias,
    ffn_bias_first,
    ffn_bias_second
  ) {

    if ((!is_first_layer) || (!prenormalization)) {
      self$attention_normalization = attention_normalization(d_token)
    }

    self$ffn_normalization = ffn_normalization(d_token)

    self$prenormalization = prenormalization

    self$attention = nn_ft_multi_head_attention(
      d_token = d_token,
      n_heads = attention_n_heads,
      dropout = attention_dropout,
      bias = attention_bias,
      initialization = attention_initialization
    )

    self$ffn = nn_ft_ffn(
      d_token = d_token,
      d_hidden = ffn_d_hidden,
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
    x_residual = self$attention(x_residual_arg,
                                      x_residual)[[1L]]
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
        attention_normalization = p_uty(init = nn_layer_norm, tags = "train"),
        ffn_d_hidden = p_dbl(lower = 1, tags = "train"),
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
mlr3pipelines::mlr_pipeops$add("nn_ft_transformer_block", PipeOpTorchFTTransformerBlock)

nn_ft_multi_head_attention = nn_module(
  "nn_ft_multi_head_attention",
  initialize = function(d_token, n_heads, dropout, bias, initialization) {
    if (n_heads > 1) {
      assert_true(d_token %% n_heads == 0)
    }
    assert_choice(initialization, c("kaiming", "xavier"))

    self$W_q = nn_linear(d_token, d_token, bias)
    self$W_k = nn_linear(d_token, d_token, bias)
    self$W_v = nn_linear(d_token, d_token, bias)
    self$W_out = if (n_heads > 1) nn_linear(d_token, d_token, bias) else NULL
    self$n_heads = n_heads
    self$dropout = if (dropout) nn_dropout(dropout) else NULL

    weights = c(self$W_q, self$W_k, self$W_v)
    for (i in seq_along(weights)) {
      m = weights[[i]]
      if (initialization == "xavier" && (i != length(weights) || !is.null(self$W_out))) {
        nn_init_xavier_uniform_(m$weight, gain = 1 / sqrt(2))
      }
      if (!is.null(m$bias)) nn_init_zeros_(m$bias)
    }
    if (!is.null(self$W_out)) nn_init_zeros_(self$W_out$bias)
  },
  reshape_ = function(input) {
    batch_size = input$shape[1]
    n_tokens = input$shape[2]
    d = input$shape[3]
    d_head = d %/% self$n_heads
    return(input$reshape(c(batch_size, n_tokens, self$n_heads, d_head))$transpose(2, 3)$reshape(c(batch_size * self$n_heads, n_tokens, d_head)))
  },
  forward = function(x_q, x_kv) {
    q = self$W_q(x_q)
    k = self$W_k(x_kv)
    v = self$W_v(x_kv)

    batch_size = q$shape[1]
    d_head_key = data.table::last(k$shape, 1) %/% self$n_heads
    d_head_value = data.table::last(v$shape, 1) %/% self$n_heads
    n_q_tokens = q$shape[2]

    q = self$reshape_(q)
    k = self$reshape_(k)
    attention_logits = torch_matmul(q, k$transpose(2, 3)) / sqrt(d_head_key)
    attention_probs = nnf_softmax(attention_logits, dim = -1)
    if (!is.null(self$dropout)) {
      attention_probs = self$dropout(attention_probs)
    }
    x = torch_matmul(attention_probs, self$reshape_(v))
    x = x$reshape(c(batch_size, self$n_heads, n_q_tokens, d_head_value))$transpose(2, 3)$reshape(c(batch_size, n_q_tokens, self$n_heads * d_head_value))
    if (!is.null(self$W_out)) x = self$W_out(x)
    return(list(x = x,
                attention_logits = attention_logits,
                attention_probs = attention_probs))
  }
)

nn_ft_ffn = nn_module(
  "nn_ft_ffn",
  initialize = function(d_token, d_hidden, bias_first, bias_second, dropout, activation) {
    # ReGLU, GeGLU activations change the size of their input
    ffn_d_hidden_multiplier = if (class(activation)[1] %in% c("nn_reglu", "nn_geglu")) 2 else 1
    self$linear_first = nn_linear(d_token, d_hidden * ffn_d_hidden_multiplier, bias_first)
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