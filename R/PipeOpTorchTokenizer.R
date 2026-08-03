initialize_token_ = function(x, d, initialization) {
  assert_choice(initialization, c("uniform", "normal"))
  d_sqrt_inv = 1 / sqrt(d)
  if (initialization == "uniform") {
    return(nn_init_uniform_(x, a = -d_sqrt_inv, b = d_sqrt_inv))
  } else if (initialization == "normal") {
    return(nn_init_normal_(x, std=d_sqrt_inv))
  } else {
    stopf("Invalid initialization: %s", initialization)
  }
}

#' @title Numeric Tokenizer
#' @inherit nn_tokenizer_num description
#' @section nn_module:
#' Calls [`nn_tokenizer_num()`] when trained where the parameter `n_features` is inferred.
#' The output shape is `(batch, n_features, d_token)`.
#'
#' @section Parameters:
#' * `d_token` :: `integer(1)`\cr
#'   The dimension of the embedding.
#' * `bias` :: `logical(1)`\cr
#'   Whether to use a bias. Is initialized to `TRUE`.
#' * `initialization` :: `character(1)`\cr
#'   The initialization method for the embedding weights. Possible values are `"uniform"` (default)
#'   and `"normal"`.
#'
#' @templateVar id nn_tokenizer_num
#' @template pipeop_torch_channels_default
#' @templateVar param_vals d_token = 10
#' @template pipeop_torch
#' @template pipeop_torch_example
#'
#' @export
PipeOpTorchTokenizerNum = R6Class("PipeOpTorchTokenizerNum",
  inherit = PipeOpTorch,
  public = list(
    #' @description Creates a new instance of this [R6][R6::R6Class] class.
    #' @template params_pipelines
    initialize = function(id = "nn_tokenizer_num", param_vals = list()) {
      param_set = ps(
        d_token = p_int(lower = 1, tags = c("train", "required")),
        bias = p_lgl(init = TRUE, tags = c("train", "required")),
        initialization = p_fct(init = "uniform", levels = c("uniform", "normal"), tags = c("train", "required"))
      )
      super$initialize(
        id = id,
        param_set = param_set,
        param_vals = param_vals,
        module_generator = nn_tokenizer_num
      )
    }
  ),
  private = list(
    .shape_dependent_params = function(shapes_in, param_vals, task) {
      c(param_vals, list(n_features = shapes_in[[1]][2]))
    },
    .shapes_out = function(shapes_in, param_vals, task) {
      if (length(shapes_in[[1]]) != 2) {
        stopf("PipeOp '%s' expects an input with 2 dimensions (batch, n_features), but got shape %s.",
          self$id, shape_to_str(shapes_in[[1L]]))
      }
      # the number of features is needed to build the module
      assert_known_dims(shapes_in[[1L]], 2L, "the feature dimension (dimension 2)", self$id)
      list(c(shapes_in[[1]], param_vals$d_token))
    }
  )
)

#' @title Numeric Tokenizer
#' @name nn_tokenizer_num
#' @description
#' Tokenizes numeric features into a dense embedding.
#' For an input of shape `(batch, n_features)` the output shape is `(batch, n_features, d_token)`.
#' @param n_features (`integer(1)`)\cr
#'   The number of features.
#' @param d_token (`integer(1)`)\cr
#'   The dimension of the embedding.
#' @param bias (`logical(1)`)\cr
#'   Whether to use a bias.
#' @param initialization (`character(1)`)\cr
#'   The initialization method for the embedding weights. Possible values are `"uniform"`
#'   and `"normal"`.
#'
#' @references
#' `r format_bib("gorishniy2021revisiting")`
#' @export
nn_tokenizer_num = nn_module(
  "nn_tokenizer_num",
  initialize = function(n_features, d_token, bias, initialization) {
    self$n_features = assert_int(n_features, lower = 1L)
    self$d_token = assert_int(d_token, lower = 1L)
    self$initialization = assert_choice(initialization, c("uniform", "normal"))
    assert_flag(bias)

    self$weight = nn_parameter(torch_empty(self$n_features, d_token))
    if (bias) {
      self$bias = nn_parameter(torch_empty(self$n_features, d_token))
    } else {
      self$bias = NULL
    }

    self$reset_parameters()
    self$n_tokens = self$weight$shape[1]
    self$d_token = self$weight$shape[2]
  },
  reset_parameters = function() {
    initialize_token_(self$weight, self$d_token, self$initialization)
    if (!is.null(self$bias)) {
      initialize_token_(self$bias, self$d_token, self$initialization)
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

#' @title Categorical Tokenizer
#' @name nn_tokenizer_categ
#' @description
#' Tokenizes categorical features into a dense embedding.
#' For an input of shape `(batch, n_features)` the output shape is `(batch, n_features, d_token)`.
#' @param cardinalities (`integer()`)\cr
#'   The number of categories for each feature.
#' @param d_token (`integer(1)`)\cr
#'   The dimension of the embedding.
#' @param bias (`logical(1)`)\cr
#'   Whether to use a bias.
#' @param initialization (`character(1)`)\cr
#'   The initialization method for the embedding weights. Possible values are `"uniform"`
#'   and `"normal"`.
#'
#' @references
#' `r format_bib("gorishniy2021revisiting")`
#'
#' @export
nn_tokenizer_categ = nn_module(
  "nn_tokenizer_categ",
  initialize = function(cardinalities, d_token, bias, initialization) {
    self$cardinalities = assert_integerish(cardinalities, lower = 1L, any.missing = FALSE,
      min.len = 1L, coerce = TRUE)
    self$d_token = assert_int(d_token, lower = 1L)

    self$initialization = assert_choice(initialization, c("uniform", "normal"))
    assert_flag(bias)
    cardinalities_cs = cumsum(cardinalities)
    category_offsets = torch_tensor(c(0, cardinalities_cs[-length(cardinalities_cs)]),
      dtype = torch_long())
    self$category_offsets = nn_buffer(category_offsets)
    n_embeddings = cardinalities_cs[length(cardinalities_cs)]

    self$embeddings = nn_embedding(n_embeddings, d_token)
    if (bias) {
      self$bias = nn_parameter(torch_empty(length(cardinalities), d_token))
    } else {
      self$bias = NULL
    }

    self$reset_parameters()
    self$n_tokens = self$category_offsets$shape[1]
    self$d_token = self$embeddings$embedding_dim
  },
  reset_parameters = function() {
    initialize_token_(self$embeddings$weight, d = self$d_token, self$initialization)
    if (!is.null(self$bias)) {
      initialize_token_(self$bias, d = self$d_token, self$initialization)
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

#' @title Categorical Tokenizer
#' @inherit nn_tokenizer_categ description
#' @section nn_module:
#' Calls [`nn_tokenizer_categ()`] when trained where the parameter `cardinalities` is inferred.
#' The output shape is `(batch, n_features, d_token)`.
#' @section Parameters:
#' * `d_token` :: `integer(1)`\cr
#'   The dimension of the embedding.
#' * `bias` :: `logical(1)`\cr
#'   Whether to use a bias. Is initialized to `TRUE`.
#' * `initialization` :: `character(1)`\cr
#'   The initialization method for the embedding weights. Possible values are `"uniform"` (default)
#'   and `"normal"`.
#' * `cardinalities` :: `integer()`\cr
#'   The number of categories for each feature.
#'   Only needs to be provided when working with [`lazy_tensor`] inputs.
#' @templateVar id nn_tokenizer_categ
#' @template pipeop_torch_channels_default
#' @templateVar param_vals d_token = 10
#' @template pipeop_torch
#' @template pipeop_torch_example
#'
#' @export
PipeOpTorchTokenizerCateg = R6Class("PipeOpTorchTokenizerCateg",
  inherit = PipeOpTorch,
  public = list(
    #' @description Creates a new instance of this [R6][R6::R6Class] class.
    #' @template params_pipelines
    initialize = function(id = "nn_tokenizer_categ", param_vals = list()) {
      param_set = ps(
        d_token = p_int(lower = 1, tags = c("train", "required")),
        bias = p_lgl(init = TRUE, tags = "train"),
        initialization = p_fct(init = "uniform", levels = c("uniform", "normal"), tags = "train"),
        cardinalities = p_uty(tags = "train", custom_check = crate(function(x) check_integerish(x, lower = 1, min.len = 1L, any.missing = FALSE, null.ok = TRUE))) # nolint
      )
      super$initialize(
        id = id,
        param_set = param_set,
        param_vals = param_vals,
        module_generator = nn_tokenizer_categ
      )
    }
  ),
  private = list(
    .shape_dependent_params = function(shapes_in, param_vals, task) {
      if ("lazy_tensor" %in% task$feature_types$type) {
        if (!single_lazy_tensor(task)) {
          stopf("Categorical tokenizer can only work with a single lazy tensor, but got %i", sum(task$feature_types$type == "lazy_tensor"))
        }
        if (is.null(param_vals$cardinalities)) {
          stopf("Categorical tokenizer received a lazy tensor input, but no parameter 'cardinalities' was specified.")
        }
        return(param_vals)
      }
      # the user may have supplied them, in which case they must not be passed twice
      if (!is.null(param_vals$cardinalities)) {
        return(param_vals)
      }
      # `$shapes_out()` rejects this too, but the module can also be built directly
      if (is.null(task)) {
        stopf("PipeOp '%s' needs to know the number of categories, but got neither a task nor the 'cardinalities' parameter.", # nolint
          self$id)
      }
      c(param_vals, list(cardinalities = unname(categ_cardinalities(task))))
    },
    .shapes_out = function(shapes_in, param_vals, task) {
      assert_ndim(shapes_in[[1L]], 2L, self$id)
      # The number of output tokens is the number of *categories* the module is built for, which
      # comes from `cardinalities` (or the task), not from the input: the embedding is indexed
      # with the category offsets, so a size-1 input dimension broadcasts against them.
      cardinalities = param_vals[["cardinalities"]]
      # the number of tokens is never unknown: without either source the module cannot be built
      # at all, so this is a configuration error rather than a shape that is merely not known yet
      if (is.null(cardinalities) && is.null(task)) {
        stopf("PipeOp '%s' needs to know the number of categories, but got neither a task nor the 'cardinalities' parameter.", # nolint
          self$id)
      }
      n_tokens = if (!is.null(cardinalities)) length(cardinalities) else length(categ_cardinalities(task))
      if (n_tokens == 0L) {
        stopf("PipeOp '%s' needs categorical features, but task '%s' has none. Either use a task with factor features or set the 'cardinalities' parameter.", # nolint
          self$id, task$id)
      }
      # the module indexes the embedding with one offset per category, so a known input width
      # that is neither 1 nor the number of categories cannot work
      n_features = shapes_in[[1L]][[2L]]
      if (!is.na(n_features) && n_features != 1L && n_features != n_tokens) {
        stopf("PipeOp '%s' was given %i cardinalities, which does not match the %i features of the input shape %s.", # nolint
          self$id, n_tokens, n_features, shape_to_str(shapes_in[[1L]]))
      }
      list(c(shapes_in[[1L]][[1L]], as.integer(n_tokens), param_vals[["d_token"]]))
    }
  )
)

#' @include aaa.R
register_po("nn_tokenizer_num", PipeOpTorchTokenizerNum)
register_po("nn_tokenizer_categ", PipeOpTorchTokenizerCateg)
