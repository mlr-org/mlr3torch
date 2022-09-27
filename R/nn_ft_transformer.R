#' @title Tabular Tokenizer
#' @description
#' Tokenizes tabular data.
#' @export

library(R6)
library(torch)
library(checkmate)
source("./R/activations.R")


#' Tabular Tokenizers
#'
#' Tokenizes tabular data.
#'
#' @param n_features (`integer(1)`)\cr
#'   The number of numeric features.
#' @param cardinalities (`integer()`)\cr
#'   The cardinalities (levels) for the factor variables.
#' @param d_token (`integer(1)`)\cr
#'   The dimension of the tokens.
#' @param bias (`logical(1)`)\cr
#'   Whether to use a bias.
#' @param cls (`logical(1)`)\cr
#'   Whether to add a cls token.
#'
#' @references `r format_bib("gorishniy2021revisiting")`
nn_tab_tokenizer = nn_module(
  "nn_tab_tokenizer",
  initialize = function(n_features, cardinalities, d_token, bias, cls) {
    self$tokenizers = list()
    assert_true(n_features > 0L || length(cardinalities) > 0L)
    if (n_features > 0L) {
      self$tokenizer_num = nn_tokenizer_numeric(n_features, d_token, bias)
    }
    if (length(cardinalities) > 0L) {
      self$tokenizer_cat = nn_tokenizer_categorical(cardinalities, d_token, bias)
    }
    if (cls) {
      self$cls = nn_cls(d_token)
    }
  },
  forward = function(input_num, input_cat) {
    tokens = list()
    if (!is.null(input_num)) {
      tokens[["x_num"]] = self$tokenizer_num(input_num)
    }
    if (!is.null(input_cat)) {
      tokens[["x_cat"]] = self$tokenizer_cat(input_cat)
    }
    tokens = torch_cat(tokens, dim = 2L)
    if (!is.null(self$cls)) { # TODO question
      tokens = self$cls(tokens)
    }
    return(tokens)
  }
)

# adapted from: https://github.com/yandex-research/rtdl/blob/main/rtdl/modules.py

# TODO: add kaiming initialization as done here: https://github.com/yandex-research/rtdl/blob/main/bin/ft_transformer.py
# Uniform initialization
initialize_token_ = function(x, d, init_type) {
  d_sqrt_inv = 1 / sqrt(d)
  if (init_type == '')
  nn_init_uniform_(x, a = -d_sqrt_inv, b = d_sqrt_inv)
}

nn_tokenizer_numeric = nn_module(
  "nn_tokenizer_numeric",
  initialize = function(n_features, d_token, bias) {
    self$n_features = checkmate::assert_integerish(n_features,
                                        lower = 1L, any.missing = FALSE, len = 1,
                                        coerce = TRUE
    )
    self$d_token = checkmate::assert_integerish(d_token,
                                     lower = 1L, any.missing = FALSE, len = 1,
                                     coerce = TRUE
    )
    checkmate::assert_flag(bias)

    self$weight = nn_parameter(torch_empty(self$n_features, d_token))
    if (bias) {
      self$bias = nn_parameter(torch_empty(self$n_features, d_token))
    } else {
      self$bias = NULL
    }

    self$reset_parameters()
  },
  reset_parameters = function() {
    initialize_token_(self$weight, self$d_token)
    if (!is.null(self$bias)) {
      initialize_token_(self$bias, self$d_token)
    }
  },
  forward = function(input) {
    x = self$weight[NULL] * input[.., NULL]
    if (!is.null(self$bias)) {
      x = x + self$bias[NULL]
    }
    return(x)
  }
)

nn_tokenizer_categorical = nn_module(
  "nn_tokenizer_categorical",
  initialize = function(cardinalities, d_token, bias) {
    self$cardinalities = assert_integerish(cardinalities,
                                           lower = 1L, any.missing = FALSE,
                                           min.len = 1L, coerce = TRUE
    )
    self$d_token = assert_integerish(d_token,
                                     lower = 1L, any.missing = FALSE, len = 1,
                                     coerce = TRUE
    )
    assert_flag(bias)
    cardinalities_cs = cumsum(cardinalities)
    category_offsets = torch_tensor(c(0, cardinalities_cs[-length(cardinalities_cs)]),
                                    dtype = torch_long()
    )
    self$register_buffer("category_offsets", category_offsets, persistent = FALSE)
    n_embeddings = cardinalities_cs[length(cardinalities_cs)]

    self$embeddings = nn_embedding(n_embeddings, d_token)
    if (bias) {
      self$bias = nn_parameter(torch_empty(length(cardinalities), d_token))
    } else {
      self$bias = NULL
    }

    self$reset_parameters()
  },
  reset_parameters = function() {
    initialize_token_(self$embeddings$weight, d = self$d_token)
    if (!is.null(self$bias)) {
      initialize_token_(self$bias, d = self$d_token)
    }
  },
  forward = function(input) {
    x = self$embeddings(input + self$category_offsets[NULL])
    if (!is.null(self$bias)) {
      x = x + self$bias[NULL]
    }
    return(x)
  }
)


nn_cls_token = nn_module(
  "nn_cls_token",
  initialize = function(d_token) {
    self$d_token = checkmate::assert_integerish(d_token,
                                                lower = 1L, any.missing = FALSE, len = 1,
                                                coerce = TRUE
    )
    self$weight = nn_parameter(torch_empty(d_token))
    self$reset_parameters()
  },
  reset_parameters = function() {
    initialize_token_(self$weight, d = self$d_token)
  },
  expand = function(...) {
    leading_dimensions = list(...)
    if(length(leading_dimensions) == 0) {
      return(self$weight)
    }
    new_dims = rep(1, length(leading_dimensions) - 1)
    return(self$weight$view(c(new_dims, -1))$expand(c(leading_dimensions, -1)))
  },
  forward = function(input) {
    return(torch_cat(list(input, self$expand(x$shape[1], 1)), dim=2)) # the length of tensor, multiplies all dimensions
  }
)


nn_reglu = nn_module(
  "nn_reglu",
  forward = function(input) {
    return(reglu(input))
  }
)


nn_geglu = nn_module(
  "nn_geglu",
  forward = function(input) {
    return(geglu(input))
  }
)


nn_multi_head_attention = nn_module(
  "nn_multi_head_attention",
  initialize = function(d_token, n_heads, dropout, bias, initialization) {
    if (n_heads > 1) {
      assert(d_token %% n_heads == 0, 'd_token must be a multiple of n_heads')
    }
    assert(initialization %in% c('kaiming', 'xavier'))

    self$W_q = nn_linear(d_token, d_token, bias)
    self$W_k = nn_linear(d_token, d_token, bias)
    self$W_v = nn_linear(d_token, d_token, bias)
    self$W_out = if (n_heads > 1) nn_linear(d_token, d_token, bias) else NULL
    self$n_heads = n_heads
    self$dropout = if (dropout) nn_dropout(dropout) else NULL

    weights = c(self$W_q, self$W_k, self$W_v)
    for (i in 1:length(weights)) {
      m = weights[[i]]
      if (initialization == 'xavier' &
          (i != length(weights) | !is.null(self$W_out))) {
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
  forward = function(x_q, x_kv, key_compression=NULL, value_compression=NULL) {
    # TODO : all or none check
    q = self$W_q(x_q)
    k = self$W_k(x_kv)
    v = self$W_v(x_kv)

    for (tensor in c(q, k, v)) {
      assert(tail(tensor$shape, 1) %% self$n_heads == 0, "Open an issue.") # TODO: what to do this
    }
    if (!is.null(key_compression)) {
      k = key_compression(k$transpose(2, 3))$transpose(2, 3)
      v = value_compression(v$transpose(2, 3))$transpose(2, 3)
    }

    batch_size = q$shape[1]
    d_head_key = tail(k$shape, 1) %/% self$n_heads
    d_head_value = tail(v$shape, 1) %/% self$n_heads
    n_q_tokens = q$shape[2]

    q = self$reshape_(q)
    k = self$reshape_(k)
    attention_logits = torch_matmul(q, k$transpose(2, 3)) / sqrt(d_head_key)
    attention_probs = nnf_softmax(attention_logits, dim=-1)
    if (!is.null(self$dropout)) {
      attention_probs = self$dropout(attention_probs)
    }
    x = torch_matmul(attention_probs, self$reshape_(v))
    x = x$reshape(c(batch_size, self$n_heads, n_q_tokens, d_head_value))$transpose(2, 3)$reshape(c(batch_size, n_q_tokens, self$n_heads * d_head_value))
    if (!is.null(self$W_out)) x = self$W_out(x)
    return(list("x" = x,
                "attention_logits" = attention_logits,
                "attention_probs" = attention_probs))
  }
)



nn_ffn = nn_module(
  "nn_ffn",
  initialize = function(d_token, d_hidden, bias_first, bias_second, dropout, activation) {
    coef = if (identical(activation, nn_geglu) | identical(activation, nn_reglu)) 2 else 1
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

