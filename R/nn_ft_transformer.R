#' @title Tabular Tokenizer
#' @description
#' Tokenizes tabular data.
#' @export

library(R6)
library(torch)
library(torchopt) # question: can i use this?
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
  initialize = function(n_features, cardinalities, d_token, bias=TRUE, cls=FALSE) {
    self$tokenizers = list()
    self$n_tokens = 0
    assert_true(n_features > 0L || length(cardinalities) > 0L)
    if (n_features > 0L) {
      self$tokenizer_num = nn_tokenizer_numeric(n_features, d_token, bias)
      self$n_tokens = self$n_tokens + self$tokenizer_num$n_tokens
    }
    if (length(cardinalities) > 0L) {
      self$tokenizer_cat = nn_tokenizer_categorical(cardinalities, d_token, bias)
      self$n_tokens = self$n_tokens + self$tokenizer_cat$n_tokens
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
initialize_token_ = function(x, d, init_type="") { #TODO init type
  d_sqrt_inv = 1 / sqrt(d)
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
    self$n_tokens = self$weight$shape[1]
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
    self$n_tokens = self$category_offsets$shape[1]
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


nn_head = nn_module(
  "nn_head",
  initialize = function(d_in, bias, activation, normalization, d_out) {
    self$normalization = normalization
    self$activation = activation
    self$linear_second = nn_linear(d_in, d_out, bias)
  },
  forward = function(x) {
    x = x[, -1]
    x = self$normalization(x)
    x = self$activation(x)
    x = self$linear(x)
    return(x)
  }
)


nn_transformer = nn_module(
  "nn_transformer",
  initialize = function(d_token,
                        n_blocks,
                        attention_n_heads,
                        attention_dropout,
                        attention_initialization,
                        attention_normalization,
                        ffn_d_hidden,
                        ffn_dropout,
                        ffn_activation,
                        ffn_normalization,
                        residual_dropout,
                        prenormalization,
                        first_prenormalization,
                        last_layer_query_idx=NULL,
                        n_tokens=NULL,
                        kv_compression_ratio=NULL,
                        kv_compression_sharing=NULL,
                        head_activation,
                        head_normalization,
                        d_out) {
    # TODO: check of last_layer_query_idx type
    if (!prenormalization) {
      assert(!first_prenormalization, "If `prenormalization` is False, then `first_prenormalization` must be False")
    }
    # TODO: check of all_or_none of [n_tokens, kv_compression_ratio, kv_compression_sharing]
    assert(kv_compression_sharing %in% c('headwise', 'key-value', 'layerwise') || is.null(kv_compression_sharing))
    if (!prenormalization) {
      warning("prenormalization is set to False. Are you sure about this? The training can become less stable. You can turn off this warning by tweaking the rtdl.Transformer.WARNINGS dictionary.")
      assert(!first_prenormalization, "If prenormalization is False, then first_prenormalization is ignored and must be set to False")
    }
    if (prenormalization && first_prenormalization) {
      warning("first_prenormalization is set to True. Are you sure about this? For example, the vanilla FTTransformer with first_prenormalization=True performs SIGNIFICANTLY worse. You can turn off this warning by tweaking the rtdl.Transformer.WARNINGS dictionary.")
    }
    if (!is.null(kv_compression_ratio) && kv_compression_sharing == 'layerwise') {
      self$shared_kv_compression = self$make_kv_compression(n_tokens, kv_compression_ratio)
    } else {
      self$shared_kv_compression = NULL
    }
    self$prenormalization = prenormalization
    self$last_layer_query_idx = last_layer_query_idx
    self$blocks = nn_module_list()
    for (layer_idx in 1:n_blocks) {
      # TODO there was no ModuleDict
      layer = list("attention" = nn_multi_head_attention(d_token=d_token,
                                                         n_heads=attention_n_heads,
                                                         dropout=attention_dropout,
                                                         bias=TRUE,
                                                         initialization=attention_initialization),
                   "ffn" = nn_ffn(d_token=d_token,
                                  d_hidden=ffn_d_hidden,
                                  bias_first=TRUE,
                                  bias_second=TRUE,
                                  dropout=ffn_dropout,
                                  activation=ffn_activation),
                   "attention_residual_dropout" = nn_dropout(residual_dropout),
                   "ffn_residual_dropout" = nn_dropout(residual_dropout),
                   "output" = nn_identity(), # for hooks-based introspection
      )
      if (layer_idx | !prenormalization | first_prenormalization) {
        layer$attention_normalization = attention_normalization(d_token) # TODO was with _make_nn_module
      }
      layer$ffn_normalization = ffn_normalization(d_token) # TODO was with _make_nn_module
      if (kv_compression_ratio & is.null(self$shared_kv_compression)) {
        layer$key_compression = make_kv_compression(n_tokens, kv_compression_ratio)
        if (kv_compression_sharing == 'headwise') {
          layer$value_compression = make_kv_compression(n_tokens, kv_compression_ratio)
        } else {
          assert(kv_compression_sharing == 'key-value', "Internal message error") # TODO error text
        }
      }
      self$blocks = append(layer, self$blocks)
    }
    self$head = nn_head(d_in=d_token,
                        d_out=d_out,
                        bias=TRUE,
                        activation=head_activation, # type: ignore
                        normalization=if (prenormalization) head_normalization else 'Identity') # TODO will this Identity work
  },
  make_kv_compression = function(n_tokens, kv_compression_ratio) {
    assert(n_tokens & kv_compression_ratio, "Internal error happened!") # TODO internal error message
    return(nn_linear(n_tokens, floor(n_tokens * kv_compression_ratio), bias=FALSE))
  },
  get_kv_compressions_ = function(layer) {
    if(!is.null(self$shared_kv_compression)) {
      result = c(self$shared_kv_compression, self$shared_kv_compression)
    } else {
      if ("key_compression" %in% names(layer) & "value_compression" %in% names(layer)) {
        result = c(layer$key_compression, layer$value_compression)
      } else {
        if ("key_compression" %in% names(layer)) {
          result = c(layer$key_compression, layer$key_compression)
        } else {
          result = c(NULL, NULL)
        }
      }
    }
  },
  start_residual_ = function(layer, stage, x) {
    assert(stage %in% c("attention", "ffn"), "Internal error") # TODO error message
    x_residual = x
    if (self$prenormalization) {
      norm_key = paste0(stage, "_normalization")
      if (norm_key %in% names(layer)) {
        x_residual = layer[[norm_key]](x_residual)
      }
    }
    return(x_residual)
  },
  end_residual_ = function(layer, stage, x, x_residual) {
    assert(stage %in% c("attention", "ffn"), "Internal error") # TODO error message
    x_residual = layer[[paste0(stage, "_residual_dropout")]](x_residual)
    x = x + x_residual
    if (!self$prenormalization) {
      x = layer[[paste0(stage, "_normalization")]](x)
    }
    return(x)
  },
  forward = function(x) {
    assert(x$ndim == 3, "The input must have 3 dimensions: (n_objects, n_tokens, d_token)")
    for (layer_idx in 1:length(self$blocks)) {
      layer = self$blocks[[layer_idx]]
      query_idx = if (layer_idx == length(self$blocks)) self$last_layer_query_idx else NULL
      x_residual = self$start_residual_(layer, 'attention', x)

      x_residual_arg = if (is.null(query_idx)) x_residual else x_residual[, query_idx]
      compressions = self$get_kv_compressions_(layer)
      x_residual_vec = layer[['attention']](x_residual_arg,
                                            x_residual,
                                            compressions[1],
                                            compressions[2])
      x = if (!is.null(query_idx)) x[, query_idx] else x
      x = self$end_residual_(layer, 'attention', x, x_residual)

      x_residual = self$start_residual_(layer, 'ffn', x)
      x_residual = layer[['ffn']](x_residual)
      x = self$end_residual_(layer, 'ffn', x, x_residual)
      x = layer[['output']](x)
    }
    x = self$head(x)
    return(x)
  }
)


get_baseline_transformer_subconfig = function() {
  parameters = list(
    'attention_n_heads'=8,
    'attention_initialization'='kaiming',
    'ffn_activation'=nn_reglu(),
    'attention_normalization'=nn_layer_norm,
    'ffn_normalization'=nn_layer_norm,
    'prenormalization'=TRUE,
    'first_prenormalization'=FALSE,
    'last_layer_query_idx'=NULL,
    'n_tokens'=NULL,
    'kv_compression_ratio'=NULL,
    'kv_compression_sharing'=NULL,
    'head_activation'=nn_relu(),
    'head_normalization'=nn_layer_norm
  )
  return(parameters)
}

get_default_transformer_config = function(n_blocks = 3) {
  assert(1 <= n_blocks && n_blocks <= 6)
  grid = list(
    'd_token'=c(96, 128, 192, 256, 320, 384),
    'attention_dropout'=c(0.1, 0.15, 0.2, 0.25, 0.3, 0.35),
    'ffn_dropout'=c(0.0, 0.05, 0.1, 0.15, 0.2, 0.25)
  )

  arch_subconfig = list()
  for (i in 1:length(grid)) {
    k = names(grid)[i]
    v = grid[[i]]
    arch_subconfig[[k]] = v[n_blocks]
  }

  baseline_subconfig = get_baseline_transformer_subconfig()
  ffn_d_hidden_factor = if (class(baseline_subconfig[['ffn_activation']])[1] %in% c("nn_reglu", "nn_geglu")) (4 / 3) else 2.0
  parameters = list(
    'n_blocks' = n_blocks,
    'residual_dropout' = 0.0,
    'ffn_d_hidden' = floor(arch_subconfig[['d_token']] * ffn_d_hidden_factor)
  )
  parameters = append(parameters, arch_subconfig)
  parameters = append(parameters, baseline_subconfig)
  return(parameters)
}

make_ = function(n_num_features, cat_cardinalities, transformer_config) {
  feature_tokenizer = nn_tab_tokenizer(n_features=n_num_features,
                                       cardinalities=cat_cardinalities,
                                       d_token=transformer_config$d_token)
  if (is.null(transformer_config$d_out)) {
    transformer_config$head_activation = NULL
  }
  if (!is.null(transformer_config$kv_compression_ratio)) {
    transformer_config$n_tokens = feature_tokenizer$n_tokens + 1
  }
  print(transformer_config)
  return(nn_ft_transformer(feature_tokenizer,
                           do.call("nn_transformer", transformer_config)
  ))
}

make_baseline = function(n_num_features,
                         cat_cardinalities,
                         d_token,
                         n_blocks,
                         attention_dropout,
                         ffn_d_hidden,
                         ffn_dropout,
                         residual_dropout,
                         last_layer_query_idx=NULL,
                         kv_compression_ratio=NULL,
                         kv_compression_sharing=NULL,
                         d_out) {
  transformer_config = get_baseline_transformer_subconfig()
  locals <- as.list(environment())
  for (arg_name in c('n_blocks',
                     'd_token',
                     'attention_dropout',
                     'ffn_d_hidden',
                     'ffn_dropout',
                     'residual_dropout',
                     'last_layer_query_idx',
                     'kv_compression_ratio',
                     'kv_compression_sharing',
                     'd_out')) {
    transformer_config[[arg_name]] = locals[[arg_name]]
  }
  return(make_(n_num_features, cat_cardinalities, transformer_config))
}

make_default = function(n_num_features,
                        cat_cardinalities,
                        n_blocks,
                        last_layer_query_idx=NULL,
                        kv_compression_ratio=NULL,
                        kv_compression_sharing=NULL,
                        d_out) {
  transformer_config = get_default_transformer_config(n_blocks=n_blocks)
  locals <- as.list(environment())
  for (arg_name in c('last_layer_query_idx',
                     'kv_compression_ratio',
                     'kv_compression_sharing',
                     'd_out')) {
    transformer_config[[arg_name]] = locals[[arg_name]]
  }
  return (make_(n_num_features, cat_cardinalities, transformer_config))
}

nn_ft_transformer = nn_module(
  "nn_ft_transformer",
  initialize = function(feature_tokenizer, transformer) {
    if (transformer$prenormalization) {
      assert(!("attention_normalization" %in% transformer$blocks[1]),
             "In the prenormalization setting, FT-Transformer does not allow using the first normalization layer in the first transformer block")
    }
    self$feature_tokenizer = feature_tokenizer
    self$cls_token = nn_cls_token(feature_tokenizer$d_token, feature_tokenizer$initialization)
    self$transformer = transformer
  },
  optimization_param_groups = function() {
    no_wd_names = c('feature_tokenizer', 'normalization', '.bias')
    # TODO two assert check missed
    needs_wd_ = function(name) {
      res = c()
      for (x in no_wd_names) {
        check_par = grepl(x, name, fixed=TRUE)
        res = c(res, !check_par)
      }
      return(all(res))
    }
    named_params = self$named_parameters()
    params_need = c()
    params_not_need = c()
    for (i in length(named_params)) {
      key = names(named_params)[i]
      val = named_params[[i]]
      if (needs_wd_(key)) {
        params_need = c(params_need, val)
      } else {
        params_not_need = c(params_not_need, val)
      }
    }
    result = c(
      list("params" = params_need),
      list("params" = params_not_need,
           "weight_decay" = 0.0)
    )
    return(result)
  },
  make_default_optimizer = function() {
    return(optim_adamw(self$optimization_param_groups(),
                       lr=1e-4,
                       weight_decay=1e-5))
  },
  forward = function(x_num, x_cat) {
    x = self$feature_tokenizer(x_num, x_cat)
    x = self$cls_token(x)
    x = self$transformer(x)
    return(x)
  }
)



