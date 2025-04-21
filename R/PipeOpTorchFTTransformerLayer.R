#' @title PipeOpTorchFTTransformerLayer
#' @description PipeOp for a single transformer layer.
PipeOpTorchFTTransformerLayer = R6::R6Class("PipeOpTorchFTTransformerLayer",
  inherit = PipeOpTorch,
  lock_objects = FALSE,
  public = list(
    #' @description Create a new instance of this [R6][R6::R6Class] class.
    #' @param id (`character(1)`)\cr
    #'   Identifier of the resulting object.
    initialize = function(id = "nn_ft_transformer_layer", param_vals = list()) {
      # TODO: double-check the parameter specification
      param_set = ps(
        d_token = p_int(lower = 1L, default = 192L),
        attention_n_heads = p_int(lower = 1L, default = 8L),
        attention_dropout = p_dbl(lower = 0, upper = 1, default = 0.2),
        attention_initialization = p_fct(levels = c("kaiming", "xavier"), default = "kaiming"),
        attention_normalization = p_uty(default = nn_layer_norm),
        ffn_d_hidden = p_int(lower = 1L, default = 256L),
        ffn_dropout = p_dbl(lower = 0, upper = 1, default = 0.1),
        ffn_activation = p_uty(),
        ffn_normalization = p_uty(default = nn_layer_norm),
        residual_dropout = p_dbl(lower = 0, upper = 1, default = 0.0),
        prenormalization = p_lgl(default = TRUE), 
        first_prenormalization = p_lgl(default = FALSE),
        is_first_layer = p_lgl(default = FALSE),
        # TODO: determine whether you can factor out last_layer_query_idx
        query_idx = p_uty(default = NULL, custom_check = function(input) check_integer(input, null.ok = TRUE)),
        last_layer_query_idx = p_uty(default = NULL, custom_check = function(input) check_integer(input, null.ok = TRUE))
      )
      
      super$initialize(
        id = id,
        module_generator = nn_ft_transformer_layer,
        param_vals = param_vals,
        param_set = param_set
      )

      self$last_layer_query_idx = param_vals$last_layer_query_idx
    }
  ),
  private = list(
    .shapes_out = function(shapes_in, param_vals, task) {
      if (is.null(param_vals$last_layer_query_idx)) {
        return(shapes_in)
      }

      if (self$last_layer_query_idx) {
        return(shapes_in[length(shapes_in)])
      }

      shapes_out = shapes_in
      shapes_out[[2]] = length(param_vals$last_layer_query_idx)
      return(shapes_out)
    }
  )
)
mlr3pipelines::mlr_pipeops$add("nn_ft_transformer_layer", PipeOpTorchFTTransformerLayer)

# TODO: remove default values from here
nn_ft_transformer_layer = nn_module(
  "nn_transformer_layer",
  initialize = function(d_token,
                        attention_n_heads,
                        attention_dropout,
                        attention_initialization,
                        ffn_d_hidden,
                        ffn_dropout,
                        ffn_activation,
                        residual_dropout,
                        prenormalization,
                        is_first_layer = FALSE,
                        first_prenormalization = FALSE,
                        attention_normalization,
                        ffn_normalization,
                        kv_compression_ratio = NULL,
                        kv_compression_sharing = NULL,
                        last_layer_query_idx,
                        query_idx) {
    self$prenormalization = prenormalization

    self$attention = nn_ft_multi_head_attention(
      d_token = d_token,
      n_heads = attention_n_heads,
      dropout = attention_dropout,
      bias = TRUE,
      initialization = attention_initialization
    )

    self$ffn = nn_ft_ffn(
      d_token = d_token,
      d_hidden = ffn_d_hidden,
      bias_first = TRUE,
      bias_second = TRUE,
      dropout = ffn_dropout,
      activation = ffn_activation
    )

    self$attention_residual_dropout = nn_dropout(residual_dropout)
    self$ffn_residual_dropout = nn_dropout(residual_dropout)

    self$output = nn_identity()
 
    # TODO: remove layer_idx and ask about how we want to handle this condition
    # layer_idx = -1
    if (!is_first_layer || !prenormalization || first_prenormalization) {
      self$attention_normalization = attention_normalization(d_token)
    }
    self$ffn_normalization = ffn_normalization(d_token)
    if (!is.null(kv_compression_ratio) && is.null(self$shared_kv_compression)) {
      self$key_compression = make_kv_compression(n_tokens, kv_compression_ratio)
      if (kv_compression_sharing == "headwise") {
        self$value_compression = make_kv_compression(n_tokens, kv_compression_ratio)
      } else {
        assert_true(kv_compression_sharing == "key_value", "kv_compression_sharing parameter should be set to either 'headwise' or 'key_value'!")
      }
    }
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

  make_kv_compression = function(n_tokens, kv_compression_ratio) {
    assert_true(n_tokens && kv_compression_ratio)
    return(nn_linear(n_tokens, floor(n_tokens * kv_compression_ratio), bias=FALSE))
  },
  get_kv_compressions_ = function() {
    if(!is.null(self$shared_kv_compression)) {
      result = c(self$shared_kv_compression, self$shared_kv_compression)
    } else {
      if ("key_compression" %in% names(self) && "value_compression" %in% names(self)) {
        result = c(self$key_compression, self$value_compression)
      } else {
        if ("key_compression" %in% names(self)) {
          result = c(self$key_compression, self$key_compression)
        } else {
          result = NULL
        }
      }
    }
    return(result)
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
    compressions = self$get_kv_compressions_()
    x_residual_vec = self$attention(x_residual_arg,
                                      x_residual,
                                      compressions[1],
                                      compressions[2])
    x = if (!is.null(self$query_idx)) x[, self$query_idx, drop = FALSE] else x
    x = self$end_residual_("attention", x, x_residual)

    x_residual = self$start_residual_("ffn", x)
    x_residual = self$ffn(x_residual)
    x = self$end_residual_("ffn", x, x_residual)

    x = self$output(x)
    return(x)
  }
)

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
    # TODO: determine whether self$W_out implementation needs to be changed
    self$W_out = if (n_heads > 1) nn_linear(d_token, d_token, bias) else NULL
    self$n_heads = n_heads
    self$dropout = if (dropout) nn_dropout(dropout) else NULL

    weights = c(self$W_q, self$W_k, self$W_v)
    for (i in seq_along(weights)) {
      m = weights[[i]]
      if (initialization == "xavier" &&
          (i != length(weights) || !is.null(self$W_out))) {
        nn_init_xavier_uniform_(m$weight, gain=1/sqrt(2))
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
  forward = function(x_q, x_kv, key_compression = NULL, value_compression = NULL) {
    assert_true(all_or_none_(key_compression, value_compression))
    q = self$W_q(x_q)
    k = self$W_k(x_kv)
    v = self$W_v(x_kv)

    if (!is.null(key_compression)) {
      k = key_compression(k$transpose(2, 3))$transpose(2, 3)
      v = value_compression(v$transpose(2, 3))$transpose(2, 3)
    }

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
    coef = if (class(activation)[1] %in% c("nn_reglu", "nn_geglu")) 2 else 1
    self$linear_first = nn_linear(d_token, d_hidden * coef, bias_first)
    self$activation = activation
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

all_or_none_ = function(...) {
  args = list(...)
  all_none = all(sapply(args, is.null))
  all_not_none = all(!sapply(args, is.null))
  return(all_none || all_not_none)
}