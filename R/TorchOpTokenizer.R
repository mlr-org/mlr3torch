TorchOpTokenizer = R6Class("TorchOpEncoder",
  inherit = TorchOp,
  public = list(
    initialize = function(id = "linear", param_vals = list()) {
      param_set = ps(
        d_token = p_int(1L, Inf, tags = "train"),
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
    .operator = "tokenizer",
    .build = function(x, param_vals, task) {

      # x is a list
      layer = torch::nn_linear(
        in_features = dim(x[[2L]]),
        out_features = params,
        bias = param_set$values$bias
      )
      return(layer)
    }
  )
)

nn_tokenizer = nn_module(
  "nn_tokenizer",
  initialize = function(n_features, cardinalities, d_token, bias) {
    self$tokenizer_num = nn_tokenizer_numeric(n_features, d_token, bias)
    self$tokenizer_cat = nn_tokenizer_categorical(cardinalities, d_token, bias)
  },
  forward = function(input) {
    tokens_num = self$tokenizer_num(input[["num"]])
    tokens_cat = self$tokenizer_cat(input[["cat"]])
    tokens = torch_cat(list(tokens_num, tokens_cat), dim = 2L)

  }
)

# adapted from: https://github.com/yandex-research/rtdl/blob/main/rtdl/modules.py

#' Uniform initialization
initialize_token_ = function(x, d) {
  d_sqrt_inv = 1 / sqrt(d)
  nn_init_uniform_(x, a = -d_sqrt_inv, b = d_sqrt_inv)
}

nn_tokenizer_numeric = nn_module(
  "nn_tokenizer_numeric",
  initialize = function(n_features, d_token, bias) {
    self$n_features = assert_integerish(n_features, lower = 1L, any.missing = FALSE, len = 1,
      coerce = TRUE
    )
    self$d_token = assert_integerish(d_token, lower = 1L, any.missing = FALSE, len = 1,
      coerce = TRUE
    )
    assert_flag(bias)

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
    self$cardinalities = assert_integerish(cardinalities, lower = 1L, any.missing = FALSE,
      min.len = 1L, coerce = TRUE)
    self$d_token = assert_integerish(d_token, lower = 1L, any.missing = FALSE, len = 1,
      coerce = TRUE)
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


#' @include mlr_torchops.R
mlr_torchops$add("tokenizer", value = TorchOpTokenizer)
# TODO: Integrate .__bobs__. into the dictionary
# .__bobs__.[["tokenizer"]] = TorchOpTokenizer$private_methods$.build
