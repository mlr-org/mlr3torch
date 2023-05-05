#' @title Tabular Tokenizer
#' @description
#' Tokenizes tabular data.
#' @export
TorchOpTabTokenizer = R6Class("TorchOpTabularTokenizer",
  inherit = TorchOp,
  public = list(
    initialize = function(id = "tab_tokenizer", param_vals = list()) {
      param_set = ps(
        d_token = p_int(1L, Inf, tags = c("train", "required")),
        bias = p_lgl(default = TRUE, tags = "train"),
        cls = p_lgl(default = FALSE, tags = "train")
      )
      param_set$values = list(bias = TRUE, cls = FALSE)
      super$initialize(
        id = id,
        param_set = param_set,
        param_vals = param_vals
      )
    }
  ),
  private = list(
    .build = function(inputs, task) {
      param_vals = self$param_set$get_values(tag = "train")
      bias = param_vals[["bias"]]
      cls = param_vals[["cls"]]
      d_token = param_vals[["d_token"]]

      n_features = sum(map_lgl(task$data(cols = task$col_roles$feature), is.numeric))
      cardinalities = Filter(function(x) !is.numeric(x), task$data(cols = task$col_roles$feature))
      cardinalities = unname(map_int(cardinalities, .f = nlevels))
      args = insert_named(param_vals, list(n_features = n_features, cardinalities = cardinalities))

      invoke(nn_tab_tokenizer, .args = args)
    }
  )
)

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
      self$tokenizer_categ = nn_tokenizer_categorical(cardinalities, d_token, bias)
    }
    if (cls) {
      self$cls = nn_cls(d_token)
    }
  },
  forward = function(input) {
    input_num = input$num
    input_categ = input$categ
    tokens = list()
    if (!is.null(input_num)) {
      tokens[["x_num"]] = self$tokenizer_num(input_num)
    }
    if (!is.null(input_categ)) {
      tokens[["x_categ"]] = self$tokenizer_cat(input_categ)
    }
    tokens = torch_cat(tokens, dim = 2L)
    if (!is.null(self$cls)) {
      tokens = self$cls(tokens)
    }
    return(tokens)
  }
)

# adapted from: https://github.com/yandex-research/rtdl/blob/main/rtdl/modules.py

# TODO: add kaiming initialization as done here: https://github.com/yandex-research/rtdl/blob/main/bin/ft_transformer.py
# Uniform initialization
initialize_token_ = function(x, d) {
  d_sqrt_inv = 1 / sqrt(d)
  nn_init_uniform_(x, a = -d_sqrt_inv, b = d_sqrt_inv)
}

nn_tokenizer_numeric = nn_module(
  "nn_tokenizer_numeric",
  initialize = function(n_features, d_token, bias) {
    self$n_features = assert_integerish(n_features,
      lower = 1L, any.missing = FALSE, len = 1,
      coerce = TRUE
    )
    self$d_token = assert_integerish(d_token,
      lower = 1L, any.missing = FALSE, len = 1,
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


#' @include mlr_torchops.R
mlr_torchops$add("tab_tokenizer", value = TorchOpTabTokenizer)
