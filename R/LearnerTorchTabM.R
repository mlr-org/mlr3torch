# ======================================================================================
# TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling (ICLR 2025)
# ======================================================================================
#
# Upstream repository : https://github.com/yandex-research/tabm
# Ported from         : the official single-file package `tabm.py`, version 0.0.3,
#                       commit 28e47ae301c92ec37787dde1ce923a0793f405b4.
#                       `paper/bin/model.py` of the same commit was used as a cross
#                       reference for the `arch_type` semantics, the loss and the
#                       prediction aggregation.
# Upstream license    : Apache License 2.0
#                       (https://github.com/yandex-research/tabm/blob/main/LICENSE)
#
# The `num_embeddings` modules are a separate port of the `rtdl_num_embeddings` package
# (MIT) and follow immediately below, under their own header.
#
# Intentional deviations from upstream:
#
#  * `share_training_batches = FALSE` is NOT supported: `forward()` only accepts
#    two-dimensional `x_num` / `x_cat`, so all k submodels always see the same batch.
#  * `activation` additionally accepts an `nn_module_generator` or a function returning an
#    `nn_module`, instead of only a name looked up in `torch`.
#  * `nn_tabm()` accepts any `nn_module` with a `get_output_shape()` method as
#    `num_embeddings`.
#  * Not ported, because `TabM` never reaches them: `BatchNorm1dEnsemble`,
#    `LayerNormEnsemble`, `MLPBackbone`, the in-place layer replacement helpers and the
#    `from_*()` constructors. `TabM.make()`'s defaults are implemented by `nn_tabm()` and the
#    learner's `.network()` instead of by a separate entry point.
#  * The loss adapter (`nn_tabm_loss`) and the probability averaging in
#    `.encode_prediction()` come from `paper/bin/model.py`; the packaged `tabm.py` contains
#    the model only.
#
# ======================================================================================

# ======================================================================================
# Embeddings for numerical features
# ("On Embeddings for Numerical Features in Tabular Deep Learning", NeurIPS 2022)
# ======================================================================================
#
# Upstream repository : https://github.com/yandex-research/rtdl-num-embeddings
# Ported from         : the single-file package `rtdl_num_embeddings.py`, version 0.0.12,
#                       commit a8fc25025c83f2321c63ff127a3bcef83bb1bfb5.
# Upstream license    : MIT
#                       (https://github.com/yandex-research/rtdl-num-embeddings/blob/main/LICENSE)
#
# Intentional deviations from upstream:
#
#  * `compute_bins()` implements the quantile-based binning only (Section 3.2.1). The
#    tree-based binning (Section 3.2.2) would need `sklearn.tree`, so the `tree_kwargs` /
#    `y` / `regression` / `verbose` arguments do not exist.
#  * Not ported: the standalone `PiecewiseLinearEncoding` (unused by TabM) and the `mask`
#    buffer that only it reads. `_PiecewiseLinearEncodingImpl` itself is ported.
#
# ======================================================================================

check_num_embeddings_input = function(x, n_features) {
  if (x$dim() < 1L) {
    stopf("The input must have at least one dimension, but has %i.", x$dim())
  }
  if (tail(x$shape, 1L) != n_features) {
    stopf("The last dimension of the input was expected to be %i, but is %i.",
      n_features, tail(x$shape, 1L))
  }
  invisible(x)
}

# Embeds each numerical feature with its own scalar linear transformation, i.e. feature `i`
# is mapped to `x[i] * weight[i, ] + bias[i, ]`: `(*, n_features)` -> `(*, n_features, d)`.
#' @examplesIf torch::torch_is_installed()
#' m = nn_linear_embeddings(3, 4)
#' m(torch::torch_randn(2, 3))$shape
nn_linear_embeddings = nn_module("nn_linear_embeddings",
  initialize = function(n_features, d_embedding) {
    self$n_features = assert_int(n_features, lower = 1L, coerce = TRUE)
    self$d_embedding = assert_int(d_embedding, lower = 1L, coerce = TRUE)
    self$weight = nn_parameter(torch_empty(self$n_features, self$d_embedding))
    self$bias = nn_parameter(torch_empty(self$n_features, self$d_embedding))
    self$reset_parameters()
  },
  reset_parameters = function() {
    d_rsqrt = self$d_embedding^(-0.5)
    nn_init_uniform_(self$weight, -d_rsqrt, d_rsqrt)
    nn_init_uniform_(self$bias, -d_rsqrt, d_rsqrt)
  },
  get_output_shape = function() {
    c(self$n_features, self$d_embedding)
  },
  forward = function(input) {
    check_num_embeddings_input(input, self$n_features)
    torch_addcmul(self$bias, self$weight, input$unsqueeze(-1L))
  }
)

# `nn_linear_embeddings()` followed by a ReLU.
#' @examplesIf torch::torch_is_installed()
#' m = nn_linear_relu_embeddings(3, 8)
#' m(torch::torch_randn(2, 3))$shape
nn_linear_relu_embeddings = nn_module("nn_linear_relu_embeddings",
  initialize = function(n_features, d_embedding = 32L) {
    self$linear = nn_linear_embeddings(n_features, d_embedding)
    self$activation = nn_relu()
  },
  get_output_shape = function() {
    self$linear$get_output_shape()
  },
  forward = function(input) {
    self$activation(self$linear(input))
  }
)

nn_periodic = nn_module("nn_periodic",
  initialize = function(n_features, k, sigma) {
    assert_number(sigma, lower = .Machine$double.eps)
    self$n_features = assert_int(n_features, lower = 1L, coerce = TRUE)
    self$k = assert_int(k, lower = 1L, coerce = TRUE)
    self$sigma = sigma
    self$weight = nn_parameter(torch_empty(self$n_features, self$k))
    self$reset_parameters()
  },
  reset_parameters = function() {
    # Unlike upstream, extreme values (~0.3% probability) are explicitly avoided.
    bound = self$sigma * 3
    nn_init_trunc_normal_(self$weight, mean = 0, std = self$sigma, a = -bound, b = bound)
  },
  forward = function(input) {
    check_num_embeddings_input(input, self$n_features)
    x = 2 * pi * self$weight * input$unsqueeze(-1L)
    torch_cat(list(torch_cos(x), torch_sin(x)), dim = -1L)
  }
)

# n separate linear layers, one per feature embedding.
nn_nlinear = nn_module("nn_nlinear",
  initialize = function(n, in_features, out_features, bias = TRUE) {
    self$n = assert_int(n, lower = 1L, coerce = TRUE)
    self$in_features = assert_int(in_features, lower = 1L, coerce = TRUE)
    self$out_features = assert_int(out_features, lower = 1L, coerce = TRUE)
    self$weight = nn_parameter(torch_empty(self$n, self$in_features, self$out_features))
    self$bias = if (bias) nn_parameter(torch_empty(self$n, self$out_features)) else NULL
    self$reset_parameters()
  },
  reset_parameters = function() {
    d_in_rsqrt = self$in_features^(-0.5)
    nn_init_uniform_(self$weight, -d_in_rsqrt, d_in_rsqrt)
    if (!is.null(self$bias)) {
      nn_init_uniform_(self$bias, -d_in_rsqrt, d_in_rsqrt)
    }
  },
  forward = function(input) {
    if (input$dim() != 3L) {
      stopf("nn_nlinear() supports only inputs with exactly one batch dimension, i.e. a shape (batch_size, n_features, d_embedding), but the input has %i dimensions.", input$dim()) # nolint
    }
    x = input$transpose(1L, 2L)
    x = torch_matmul(x, self$weight)
    x = x$transpose(1L, 2L)
    if (!is.null(self$bias)) x = x + self$bias
    x
  }
)

#' @title Periodic Embeddings for Numerical Features
#' @name nn_periodic_embeddings
#'
#' @description
#' Embeddings for numerical features based on periodic activations, i.e. the *PLR*
#' embeddings of the paper: each feature is passed through `cos`/`sin` of `n_frequencies`
#' learned frequencies, followed by a linear layer and (optionally) a ReLU.
#' For an input of shape `(*, n_features)` the output shape is
#' `(*, n_features, d_embedding)`.
#'
#' @param n_features (`integer(1)`)\cr
#'   The number of numerical features.
#' @param d_embedding (`integer(1)`)\cr
#'   The embedding size. Default is `24`.
#' @param n_frequencies (`integer(1)`)\cr
#'   The number of frequencies of each feature (`k` in Section 3.3 of the paper).
#'   Default is `48`.
#' @param frequency_init_scale (`numeric(1)`)\cr
#'   The initialization scale of the frequencies (`sigma` in Section 3.3 of the paper).
#'   This is an important hyperparameter. Default is `0.01`.
#' @param activation (`logical(1)`)\cr
#'   Whether to apply the ReLU activation. Must be `TRUE` if `lite` is `TRUE`.
#'   Default is `TRUE`.
#' @param lite (`logical(1)`)\cr
#'   If `TRUE`, the outer linear layer is shared between all features (the variant
#'   introduced by the TabR paper). Has no default upstream.
#'
#' @references
#' `r format_bib("gorishniy2022embeddings")`
#'
#' @noRd
#' @examplesIf torch::torch_is_installed()
#' m = nn_periodic_embeddings(3, 8, lite = FALSE)
#' m(torch::torch_randn(2, 3))$shape
nn_periodic_embeddings = nn_module("nn_periodic_embeddings",
  initialize = function(n_features, d_embedding = 24L, n_frequencies = 48L,
    frequency_init_scale = 0.01, activation = TRUE, lite) {
    assert_flag(activation)
    assert_flag(lite)
    n_frequencies = assert_int(n_frequencies, lower = 1L, coerce = TRUE)
    d_embedding = assert_int(d_embedding, lower = 1L, coerce = TRUE)
    self$periodic = nn_periodic(n_features, n_frequencies, frequency_init_scale)
    self$linear = if (lite) {
      # The lite variation was introduced by the TabR paper.
      if (!activation) {
        stopf("lite = TRUE is allowed only when activation = TRUE.")
      }
      nn_linear(2L * n_frequencies, d_embedding)
    } else {
      nn_nlinear(n_features, 2L * n_frequencies, d_embedding)
    }
    self$activation = if (activation) nn_relu() else NULL
    self$out_shape = c(self$periodic$n_features, d_embedding)
  },
  get_output_shape = function() {
    self$out_shape
  },
  forward = function(input) {
    x = self$periodic(input)
    x = self$linear(x)
    if (!is.null(self$activation)) x = self$activation(x)
    x
  }
)

check_bins = function(bins) {
  assert_list(bins, min.len = 1L, types = "torch_tensor",
    .var.name = "bins (a list of torch tensors)")
  for (i in seq_along(bins)) {
    b = bins[[i]]
    if (b$dim() != 1L) {
      stopf("Each element of `bins` must have exactly one dimension, but element %i has %i.", i, b$dim())
    }
    if (b$shape[1L] < 2L) {
      stopf("All features must have at least two bin edges, but feature %i has %i.", i, b$shape[1L])
    }
    if (!as.logical(b$isfinite()$all())) {
      stopf("Bin edges must not contain NaN/Inf/-Inf, but those of feature %i do.", i)
    }
    if (b$shape[1L] > 2L && as.logical((b$narrow(1L, 1L, b$shape[1L] - 1L) >= b$narrow(1L, 2L, b$shape[1L] - 1L))$any())) { # nolint
      stopf("Bin edges must be sorted, but those of feature %i are not.", i)
    }
  }
  invisible(bins)
}

#' @title Compute Bin Edges for Piecewise-Linear Embeddings
#' @name compute_bins
#'
#' @description
#' Computes the quantile-based bin edges (Section 3.2.1 of the paper) that
#' [`nn_piecewise_linear_embeddings()`] expects.
#' The bins must be computed on the **training data**.
#'
#' The tree-based binning of Section 3.2.2 of the paper is not implemented, because it
#' requires fitting decision trees, which would add a package dependency.
#'
#' @param x ([`torch_tensor`][torch::torch_tensor], `matrix()` or `data.frame()`)\cr
#'   The training data of the numerical features, of shape `(n, n_features)`.
#' @param n_bins (`integer(1)`)\cr
#'   The number of bins. Must be larger than 1 and smaller than the number of rows of
#'   `x`. Default is `48`.
#'
#' @return A `list()` of one-dimensional [`torch_tensor`][torch::torch_tensor]s with the
#'   bin edges of each feature. A feature has at most `n_bins + 1` edges (fewer if some
#'   quantiles coincide).
#'
#' @references
#' `r format_bib("gorishniy2022embeddings")`
#'
#' @noRd
#' @examplesIf torch::torch_is_installed()
#' bins = compute_bins(matrix(rnorm(200), ncol = 2), n_bins = 4)
#' lengths(lapply(bins, as.numeric))
compute_bins = function(x, n_bins = 48L) {
  if (!inherits(x, "torch_tensor")) {
    x = torch_tensor(as.matrix(x), dtype = torch_float())
  }
  n_bins = assert_int(n_bins, lower = 2L, coerce = TRUE)
  if (x$dim() != 2L) {
    stopf("`x` must have exactly two dimensions, but has %i.", x$dim())
  }
  n = x$shape[1L]
  p = x$shape[2L]
  if (n < 2L) {
    stopf("`x` must have at least two rows, but has %i.", n)
  }
  if (p < 1L) {
    stopf("`x` must have at least one column.")
  }
  if (!as.logical(x$isfinite()$all())) {
    stopf("`x` must not contain NaN/Inf/-Inf.")
  }
  constant = as.logical((x == x[1, ])$all(dim = 1L))
  if (any(constant)) {
    stopf("All columns of `x` must have at least two distinct values, but column(s) %s do not.",
      paste(which(constant), collapse = ", "))
  }
  if (n_bins >= n) {
    stopf("`n_bins` must be smaller than the number of rows of `x`, but n_bins = %i and nrow = %i.", n_bins, n) # nolint
  }

  # Removing identical quantiles after computing them is not the same
  # as limiting the number of quantiles by the number of distinct values.
  quantiles = torch_quantile(x, torch_linspace(0, 1, n_bins + 1L)$to(dtype = x$dtype), dim = 1L)
  bins = lapply(seq_len(p), function(j) {
    # the quantiles are sorted, so consecutive uniqueness is the same as uniqueness
    torch_unique_consecutive(quantiles[, j])[[1L]]
  })
  check_bins(bins)
  bins
}

nn_piecewise_linear_encoding_impl = nn_module("nn_piecewise_linear_encoding_impl",
  initialize = function(bins) {
    check_bins(bins)
    n_features = length(bins)
    n_bins = map_int(bins, function(b) b$shape[1L] - 1L)
    max_n_bins = max(n_bins)

    weight = torch_zeros(n_features, max_n_bins)
    bias = torch_zeros(n_features, max_n_bins)
    for (i in seq_len(n_features)) {
      edges = bins[[i]]
      n_edges = edges$shape[1L]
      # The piecewise-linear encoding of one feature is
      # `[1, ..., 1, (x - this_bin_left_edge) / this_bin_width, 0, ..., 0]`; weight and
      # bias implement the expression in the middle, before the clipping to [0, 1].
      bin_width = torch_diff(edges)
      w = 1 / bin_width
      b = -edges$narrow(1L, 1L, n_edges - 1L) / bin_width
      # The last encoding component is always stored in the last column, so that the
      # clamping can be applied to all features at once.
      weight[i, max_n_bins] = w[n_bins[i]]
      bias[i, max_n_bins] = b[n_bins[i]]
      if (n_bins[i] > 1L) {
        weight[i, 1:(n_bins[i] - 1L)] = w[1:(n_bins[i] - 1L)]
        bias[i, 1:(n_bins[i] - 1L)] = b[1:(n_bins[i] - 1L)]
      }
      # Everything in between stays zero.
    }
    self$weight = nn_buffer(weight)
    self$bias = nn_buffer(bias)
    self$max_n_bins = max_n_bins

    single_bin_mask = torch_tensor(n_bins == 1L, dtype = torch_bool())
    self$single_bin_mask = if (any(n_bins == 1L)) nn_buffer(single_bin_mask) else NULL
  },
  get_max_n_bins = function() {
    self$max_n_bins
  },
  forward = function(input) {
    # (batch, n_features) -> (batch, n_features, max_n_bins)
    x = torch_addcmul(self$bias, self$weight, input$unsqueeze(-1L))
    n = self$max_n_bins
    if (n > 1L) {
      # NOTE: `$narrow()` is deliberately avoided here; passing an R integer as its
      # `start` argument intermittently produced a corrupted index ("start out of
      # range ... but got <garbage>") inside a `jit_trace()`d graph.
      parts = list(x[, , 1:1]$clamp(max = 1))
      if (n > 2L) {
        parts[[length(parts) + 1L]] = x[, , 2:(n - 1L)]$clamp(0, 1)
      }
      last = x[, , n:n]
      parts[[length(parts) + 1L]] = if (is.null(self$single_bin_mask)) {
        last$clamp(min = 0)
      } else {
        # For features with a single bin the encoding behaves like min-max scaling.
        torch_where(self$single_bin_mask$unsqueeze(-1L), last, last$clamp(min = 0))
      }
      x = torch_cat(parts, dim = -1L)
    }
    x
  }
)

#' @title Piecewise-Linear Embeddings for Numerical Features
#' @name nn_piecewise_linear_embeddings
#'
#' @description
#' The piecewise-linear embeddings of the paper: each numerical feature is first encoded
#' by the piecewise-linear encoding defined by its bin edges (see [`compute_bins()`]) and
#' then embedded with a per-feature linear layer.
#' For an input of shape `(batch, n_features)` the output shape is
#' `(batch, n_features, d_embedding)`.
#'
#' @param bins (`list()` of [`torch_tensor`][torch::torch_tensor])\cr
#'   The bin edges, as computed by [`compute_bins()`] on the training data.
#' @param d_embedding (`integer(1)`)\cr
#'   The embedding size.
#' @param activation (`logical(1)`)\cr
#'   Whether to apply a ReLU activation in the end.
#'
#' @references
#' `r format_bib("gorishniy2022embeddings", "gorishniy2025tabm")`
#'
#' @noRd
#' @examplesIf torch::torch_is_installed()
#' bins = compute_bins(matrix(rnorm(200), ncol = 2), n_bins = 4)
#' m = nn_piecewise_linear_embeddings(bins, d_embedding = 8, activation = FALSE)
#' m(torch::torch_randn(3, 2))$shape
nn_piecewise_linear_embeddings = nn_module("nn_piecewise_linear_embeddings",
  initialize = function(bins, d_embedding, activation) {
    d_embedding = assert_int(d_embedding, lower = 1L, coerce = TRUE)
    assert_flag(activation)
    check_bins(bins)

    n_features = length(bins)
    self$linear0 = nn_linear_embeddings(n_features, d_embedding)
    self$impl = nn_piecewise_linear_encoding_impl(bins)
    # the bias is already part of linear0
    self$linear = nn_nlinear(n_features, self$impl$get_max_n_bins(), d_embedding, bias = FALSE)
    # zero-initialized, so the whole embedding behaves like a linear embedding at
    # initialization and the piecewise-linear component is learnt incrementally
    nn_init_zeros_(self$linear$weight)
    self$activation = if (activation) nn_relu() else NULL
    self$out_shape = c(n_features, d_embedding)
  },
  get_output_shape = function() {
    self$out_shape
  },
  forward = function(input) {
    if (input$dim() != 2L) {
      stopf("nn_piecewise_linear_embeddings() only supports inputs with exactly one batch dimension, but the input has %i dimensions.", input$dim()) # nolint
    }
    x_linear = self$linear0(input)
    x_ple = self$linear(self$impl(input))
    if (!is.null(self$activation)) x_ple = self$activation(x_ple)
    x_linear + x_ple
  }
)

# --------------------------------------------------------------------------------------
# Initialization (upstream section "Initialization")
# --------------------------------------------------------------------------------------

tabm_init_rsqrt_uniform_ = function(tensor, d) {
  d_rsqrt = d^(-0.5)
  nn_init_uniform_(tensor, -d_rsqrt, d_rsqrt)
}

tabm_init_random_signs_ = function(tensor) {
  with_no_grad(tensor$bernoulli_(0.5)$mul_(2)$add_(-1))
  tensor
}

# `distribution` is one of "ones", "normal", "random-signs".
# `chunks` (upstream: `chunks`) splits the last dimension of `x` into consecutive
# blocks; all values within one block share the same randomly drawn value.
tabm_init_scaling_ = function(x, distribution, chunks = NULL) {
  init_fn = switch(distribution,
    "ones" = nn_init_ones_,
    "normal" = nn_init_normal_,
    "random-signs" = tabm_init_random_signs_,
    stopf("Unknown scaling initialization '%s'.", distribution)
  )
  if (distribution == "ones" && !is.null(chunks)) {
    stopf("When the scaling initialization is 'ones', chunks must be NULL.")
  }
  if (is.null(chunks)) {
    return(init_fn(x))
  }
  shape = dim(x)
  if (sum(chunks) != shape[length(shape)]) {
    stopf("The tensor shape (%i) and the chunks (sum: %i) are incompatible.",
      shape[length(shape)], sum(chunks))
  }
  leading = shape[-length(shape)]
  with_no_grad({
    chunk_start = 0L
    for (chunk_size in chunks) {
      value = init_fn(do.call(torch_empty, as.list(c(leading, 1L))))
      x[.., (chunk_start + 1L):(chunk_start + chunk_size)] = value
      chunk_start = chunk_start + chunk_size
    }
  })
  x
}

# --------------------------------------------------------------------------------------
# Basic modules (upstream section "Basics modules")
# --------------------------------------------------------------------------------------

# Deviation: the input codes are 1-based, i.e. the i-th feature takes values in
# `1:cardinalities[i]` (upstream: `0:(cardinalities[i] - 1)`).
nn_tabm_one_hot = nn_module("nn_tabm_one_hot",
  initialize = function(cardinalities) {
    self$cardinalities = assert_integerish(cardinalities, lower = 1L, any.missing = FALSE,
      min.len = 1L, coerce = TRUE)
  },
  forward = function(input) {
    cards = self$cardinalities
    # without this, additional columns would be dropped silently, because only the first
    # `length(cards)` are read
    if (input$shape[length(input$shape)] != length(cards)) {
      stopf("Expected %i categorical features, but got %i.", length(cards),
        input$shape[length(input$shape)])
    }
    torch_cat(lapply(seq_along(cards), function(i) {
      nnf_one_hot(input[, i], num_classes = cards[i])
    }), dim = -1L)$to(dtype = torch_float())
  }
)

nn_tabm_elementwise_affine = nn_module("nn_tabm_elementwise_affine",
  initialize = function(shape, bias, scaling_init, scaling_init_chunks = NULL) {
    self$scaling_init = scaling_init
    self$scaling_init_chunks = scaling_init_chunks
    self$weight = nn_parameter(do.call(torch_empty, as.list(shape)))
    self$bias = if (bias) nn_parameter(do.call(torch_empty, as.list(shape))) else NULL
    self$reset_parameters()
  },
  reset_parameters = function() {
    tabm_init_scaling_(self$weight, self$scaling_init, self$scaling_init_chunks)
    if (!is.null(self$bias)) {
      nn_init_zeros_(self$bias)
    }
  },
  forward = function(input) {
    if (is.null(self$bias)) input * self$weight else input * self$weight + self$bias
  }
)

# --------------------------------------------------------------------------------------
# Ensemble modules (upstream section "Ensemble modules")
# --------------------------------------------------------------------------------------

# Turns `(batch, d)` into `(batch, k, d)` without copying.
tabm_ensemble_view = function(x, k) {
  if (x$dim() != 2L) {
    stopf("The input must have two dimensions, but has %i.", x$dim())
  }
  x$unsqueeze(2L)$expand(c(-1L, k, -1L))
}

nn_tabm_ensemble_view = nn_module("nn_tabm_ensemble_view",
  initialize = function(k) {
    self$k = assert_int(k, lower = 1L, coerce = TRUE)
  },
  forward = function(input) {
    tabm_ensemble_view(input, self$k)
  }
)

# k independent linear layers applied to k inputs.
nn_tabm_linear_ensemble = nn_module("nn_tabm_linear_ensemble",
  initialize = function(in_features, out_features, bias = TRUE, k) {
    self$in_features = assert_int(in_features, lower = 1L, coerce = TRUE)
    self$out_features = assert_int(out_features, lower = 1L, coerce = TRUE)
    self$k = assert_int(k, lower = 1L, coerce = TRUE)
    self$weight = nn_parameter(torch_empty(self$k, self$in_features, self$out_features))
    self$bias = if (bias) nn_parameter(torch_empty(self$k, self$out_features)) else NULL
    self$reset_parameters()
  },
  reset_parameters = function() {
    tabm_init_rsqrt_uniform_(self$weight, self$in_features)
    if (!is.null(self$bias)) {
      tabm_init_rsqrt_uniform_(self$bias, self$in_features)
    }
  },
  forward = function(input) {
    x = input$transpose(1L, 2L)
    x = torch_matmul(x, self$weight)
    x = x$transpose(1L, 2L)
    if (!is.null(self$bias)) x = x + self$bias
    x
  }
)

# equation (5) of the BatchEnsemble paper with the
# TabM-specific initialization options for the R and S matrices.
nn_tabm_linear_batch_ensemble = nn_module("nn_tabm_linear_batch_ensemble",
  initialize = function(in_features, out_features, bias = TRUE, k, scaling_init,
    first_scaling_init_chunks = NULL) {
    self$in_features = assert_int(in_features, lower = 1L, coerce = TRUE)
    self$out_features = assert_int(out_features, lower = 1L, coerce = TRUE)
    self$k = assert_int(k, lower = 1L, coerce = TRUE)
    assert_character(scaling_init, min.len = 1L, max.len = 2L, any.missing = FALSE)
    self$first_scaling_init = scaling_init[[1L]]
    self$second_scaling_init = scaling_init[[length(scaling_init)]]
    self$first_scaling_init_chunks = first_scaling_init_chunks

    self$weight = nn_parameter(torch_empty(self$out_features, self$in_features))
    self$r = nn_parameter(torch_empty(self$k, self$in_features))
    self$s = nn_parameter(torch_empty(self$k, self$out_features))
    self$bias = if (bias) nn_parameter(torch_empty(self$k, self$out_features)) else NULL
    self$reset_parameters()
  },
  reset_parameters = function() {
    tabm_init_rsqrt_uniform_(self$weight, self$in_features)
    tabm_init_scaling_(self$r, self$first_scaling_init, self$first_scaling_init_chunks)
    tabm_init_scaling_(self$s, self$second_scaling_init, NULL)
    if (!is.null(self$bias)) {
      # All k biases share the same initialization.
      bias_init = tabm_init_rsqrt_uniform_(torch_empty(self$out_features), self$in_features)
      with_no_grad(self$bias$copy_(bias_init$expand(c(self$k, self$out_features))))
    }
  },
  forward = function(input) {
    x = input * self$r
    x = torch_matmul(x, self$weight$t())
    x = x * self$s
    if (!is.null(self$bias)) x = x + self$bias
    x
  }
)

# --------------------------------------------------------------------------------------
# MLP backbones (upstream section "MLP modules")
# --------------------------------------------------------------------------------------

# Upstream resolves `activation` via `getattr(torch.nn, activation)`. Here, in addition
# to a name, a module generator (e.g. `nn_relu`) or any function returning an `nn_module`
# is accepted. A fresh module is constructed on every call, because each block needs
# its own activation instance.
tabm_activation = function(activation) {
  if (is.function(activation)) {
    module = activation()
    if (!inherits(module, "nn_module")) {
      stopf("The `activation` function must return an `nn_module`, but it returned an object of class '%s'.", class(module)[[1L]]) # nolint
    }
    return(module)
  }
  if (!test_string(activation, min.chars = 1L)) {
    stopf("`activation` must be a `character(1)`, an `nn_module_generator` or a function returning an `nn_module`, but is of class '%s'.", class(activation)[[1L]]) # nolint
  }
  ns = asNamespace("torch")
  get_generator = function(nm) {
    if (!exists(nm, envir = ns, inherits = FALSE)) {
      return(NULL)
    }
    generator = get(nm, envir = ns)
    if (inherits(generator, "nn_module_generator")) generator else NULL
  }
  for (nm in unique(c(activation, paste0("nn_", activation), paste0("nn_", tolower(activation))))) {
    generator = get_generator(nm)
    if (!is.null(generator)) {
      return(tabm_activation(generator))
    }
  }
  stopf("Cannot resolve the activation '%s'. Provide the name of an activation of the torch package (e.g. \"relu\", \"nn_relu\" or \"ReLU\"), an `nn_module_generator` (e.g. `nn_relu`), or a function returning an `nn_module`.", activation) # nolint
}

# `make_linear(index, in_features, out_features)`
# is the `_make_linear()` hook of the respective subclass (index is 1-based here).
tabm_make_blocks = function(d_in, n_blocks, d_block, dropout, activation, make_linear) {
  assert_int(d_in, lower = 1L)
  assert_int(n_blocks, lower = 1L)
  assert_int(d_block, lower = 1L)
  nn_module_list(lapply(seq_len(n_blocks), function(i) {
    nn_sequential(
      make_linear(i, if (i == 1L) d_in else d_block, d_block),
      tabm_activation(activation),
      nn_dropout(dropout)
    )
  }))
}

# Used by arch_type "tabm-mini".
nn_tabm_mlp_backbone_mini_ensemble = nn_module("nn_tabm_mlp_backbone_mini_ensemble",
  initialize = function(d_in, n_blocks, d_block, dropout, activation = "relu", k,
    affine_bias, affine_scaling_init, affine_scaling_init_chunks = NULL) {
    self$n_blocks = n_blocks
    self$k = k
    self$d_out = d_block
    self$blocks = tabm_make_blocks(d_in, n_blocks, d_block, dropout, activation,
      # The same linear layer is used by all k backbones.
      function(index, in_features, out_features) nn_linear(in_features, out_features))
    self$affine = nn_tabm_elementwise_affine(
      shape = c(k, d_in),
      bias = affine_bias,
      scaling_init = affine_scaling_init,
      scaling_init_chunks = affine_scaling_init_chunks
    )
  },
  forward = function(input) {
    x = self$affine(input)
    for (i in seq_len(self$n_blocks)) {
      x = self$blocks[[i]](x)
    }
    x
  }
)

# Used by arch_type "tabm".
nn_tabm_mlp_backbone_batch_ensemble = nn_module("nn_tabm_mlp_backbone_batch_ensemble",
  initialize = function(d_in, n_blocks, d_block, dropout, activation = "relu", k,
    tabm_init, scaling_init, start_scaling_init_chunks = NULL) {
    self$n_blocks = n_blocks
    self$k = k
    self$d_out = d_block
    self$blocks = tabm_make_blocks(d_in, n_blocks, d_block, dropout, activation,
      function(index, in_features, out_features) {
        nn_tabm_linear_batch_ensemble(
          in_features, out_features, k = k,
          scaling_init = if (tabm_init) {
            if (index == 1L) c(scaling_init, "ones") else "ones"
          } else {
            scaling_init
          },
          first_scaling_init_chunks = if (index == 1L) start_scaling_init_chunks else NULL
        )
      })
  },
  forward = function(input) {
    x = input
    for (i in seq_len(self$n_blocks)) {
      x = self$blocks[[i]](x)
    }
    x
  }
)

tabm_make_backbone = function(d_in, n_blocks, d_block, dropout, activation, k, arch_type,
  start_scaling_init, start_scaling_init_chunks) {
  if (is.null(start_scaling_init)) {
    stopf("When arch_type is '%s', start_scaling_init must not be NULL.", arch_type)
  }

  switch(arch_type,
    "tabm" = nn_tabm_mlp_backbone_batch_ensemble(
      d_in = d_in, n_blocks = n_blocks, d_block = d_block, dropout = dropout,
      activation = activation, k = k, tabm_init = TRUE,
      scaling_init = start_scaling_init,
      start_scaling_init_chunks = start_scaling_init_chunks
    ),
    "tabm-mini" = nn_tabm_mlp_backbone_mini_ensemble(
      d_in = d_in, n_blocks = n_blocks, d_block = d_block, dropout = dropout,
      activation = activation, k = k, affine_bias = FALSE,
      affine_scaling_init = start_scaling_init,
      affine_scaling_init_chunks = start_scaling_init_chunks
    ),
    stopf("Unknown arch_type '%s'.", arch_type)
  )
}

# --------------------------------------------------------------------------------------
# The TabM module
# --------------------------------------------------------------------------------------

#' @title TabM Network
#'
#' @description
#' TabM -- a tabular deep learning model that makes **M**ultiple predictions.
#' One `nn_tabm` efficiently represents an ensemble of `k` MLPs that are trained in
#' parallel and that share most of their weights.
#'
#' Numerical features enter the network unchanged, or, if `num_embeddings` is given, are
#' first embedded feature-wise (see [`nn_linear_relu_embeddings()`],
#' [`nn_periodic_embeddings()`] and [`nn_piecewise_linear_embeddings()`]).
#' Categorical features are one-hot encoded (their integer codes must be 1-based, which
#' is what [`batchgetter_categ()`] produces).
#' The concatenated flat representation is then processed by `k` (mostly shared) MLP
#' backbones.
#'
#' For an input of shape `(batch, n_num_features)` / `(batch, n_cat_features)` the output
#' shape is `(batch, k, d_out)`, i.e. one prediction per ensemble member.
#'
#' @section Ensemble Output:
#' Because the output contains the `k` predictions of the ensemble members, it cannot be
#' fed into a standard loss function, and it must be aggregated before it can be
#' interpreted as a prediction.
#' [`LearnerTorchTabM`] (`lrn("classif.tabm")` / `lrn("regr.tabm")`) takes care of both.
#' `nn_tabm` can also be plugged into [`LearnerTorchModule`] (`lrn("classif.module")`) for
#' training, in which case the loss has to be wrapped with `nn_tabm_loss()`. Predicting with
#' such a learner does *not* work out of the box, because the default prediction encoder
#' expects a `(batch, d_out)` output and would interpret the `k` dimension as additional
#' observations. Aggregating over the ensemble members requires a custom prediction encoder;
#' use [`LearnerTorchTabM`] if you just want predictions.
#'
#' @param task ([`Task`][mlr3::Task] or `NULL`)\cr
#'   If provided, `n_num_features`, `cat_cardinalities` and `d_out` are inferred from
#'   the task (unless they are given explicitly). This makes it possible to use
#'   `nn_tabm` with [`LearnerTorchModule`].
#' @param n_num_features (`integer(1)`)\cr
#'   The number of numerical features.
#' @param cat_cardinalities (`integer()` or `NULL`)\cr
#'   The number of categories of each categorical feature.
#' @param d_out (`integer(1)` or `NULL`)\cr
#'   The output dimension. If `NULL`, the output of the `k` backbones is returned.
#' @param num_embeddings ([`nn_module`][torch::nn_module] or `NULL`)\cr
#'   Embeddings for the numerical features, applied before the backbone and shared
#'   between the `k` submodels. Must provide a `get_output_shape()` method returning
#'   `c(n_num_features, d_embedding)`; [`nn_linear_relu_embeddings()`],
#'   [`nn_periodic_embeddings()`] and [`nn_piecewise_linear_embeddings()`] (with
#'   do. If `NULL` (default), the numerical features enter the backbone
#'   unchanged.
#' @param arch_type (`character(1)`)\cr
#'   One of `"tabm"` (default) or `"tabm-mini"`.
#' @param k (`integer(1)`)\cr
#'   The number of ensemble members.
#' @param n_blocks (`integer(1)`)\cr
#'   The number of blocks (depth) of the MLP backbone.
#'   If `NULL`, `2` is used when `num_embeddings` is given and `3` otherwise.
#' @param d_block (`integer(1)`)\cr
#'   The width of the MLP backbone.
#' @param dropout (`numeric(1)`)\cr
#'   The dropout rate.
#' @param activation (`character(1)`, `nn_module_generator` or `function`)\cr
#'   The activation function. Either the name of an activation of the `torch` package
#'   (e.g. `"relu"`, `"nn_relu"` or `"ReLU"`), an
#'   [`nn_module_generator`][torch::nn_module] such as [`nn_relu`][torch::nn_relu], or a
#'   function returning an [`nn_module`][torch::nn_module]. Default is `"relu"`.
#' @param start_scaling_init (`character(1)` or `NULL`)\cr
#'   The initialization of the very first (non-shared) scaling, either `"random-signs"`
#'   or `"normal"`.
#'   If `NULL` otherwise, `"normal"` is used when `num_embeddings` is given and
#'   `"random-signs"` otherwise (this is upstream's `TabM.make()` heuristic).
#'
#' @references
#' `r format_bib("gorishniy2025tabm", "wen2020batchensemble")`
#'
#' @noRd
#' @examplesIf torch::torch_is_installed()
#' net = nn_tabm(n_num_features = 4, cat_cardinalities = c(3, 2), d_out = 3,
#'   k = 4, n_blocks = 2, d_block = 8, dropout = 0.1)
#' x_num = torch::torch_randn(5, 4)
#' x_cat = torch::torch_stack(list(
#'   torch::torch_randint(1, 3, 5, dtype = torch::torch_long()),
#'   torch::torch_randint(1, 2, 5, dtype = torch::torch_long())
#' ), dim = 2)
#' net(x_num = x_num, x_cat = x_cat)$shape
#'
#' # with periodic embeddings for the numerical features
#' net = nn_tabm(n_num_features = 4, d_out = 3, k = 4, n_blocks = 2, d_block = 8,
#'   num_embeddings = nn_periodic_embeddings(4, d_embedding = 6, lite = FALSE))
#' net(x_num = x_num)$shape
nn_tabm = nn_module("nn_tabm",
  initialize = function(task = NULL, n_num_features = NULL, cat_cardinalities = NULL,
    d_out = NULL, num_embeddings = NULL, arch_type = "tabm", k = 32L, n_blocks = NULL,
    d_block = 512L, dropout = 0.1, activation = "relu", start_scaling_init = NULL) {
    if (!is.null(task)) {
      assert_class(task, "Task")
      if (is.null(n_num_features)) n_num_features = n_num_features(task)
      if (is.null(cat_cardinalities)) cat_cardinalities = unname(categ_cardinalities(task))
      if (is.null(d_out)) d_out = output_dim_for(task)
    }
    n_num_features = assert_int(n_num_features %??% 0L, lower = 0L, coerce = TRUE)
    cat_cardinalities = assert_integerish(cat_cardinalities %??% integer(0),
      lower = 1L, any.missing = FALSE, coerce = TRUE)
    d_out = assert_int(d_out, lower = 1L, null.ok = TRUE, coerce = TRUE)
    arch_type = assert_choice(arch_type, c("tabm", "tabm-mini"))
    k = assert_int(k, lower = 1L, coerce = TRUE)
    assert_number(dropout, lower = 0, upper = 1)
    assert_choice(start_scaling_init, c("random-signs", "normal"), null.ok = TRUE)
    assert_class(num_embeddings, "nn_module", null.ok = TRUE)

    if (n_num_features == 0L && !length(cat_cardinalities)) {
      stopf("nn_tabm() requires at least one numerical or one categorical feature.")
    }

    # Representation sizes of all features (upstream: `d_features`), which double as the
    # initialization chunks of the very first scaling.
    d_features = if (is.null(num_embeddings)) {
      rep(1L, n_num_features)
    } else {
      if (n_num_features == 0L) {
        stopf("nn_tabm() received `num_embeddings`, but there are no numerical features.")
      }
      if (!is.function(num_embeddings$get_output_shape)) {
        stopf("The `num_embeddings` module must provide a `get_output_shape()` method.")
      }
      shape = num_embeddings$get_output_shape()
      if (shape[[1L]] != n_num_features) {
        stopf("The `num_embeddings` module was created for %i features, but n_num_features is %i.", shape[[1L]], n_num_features) # nolint
      }
      rep(as.integer(shape[[2L]]), n_num_features)
    }
    d_features = c(d_features, cat_cardinalities)

    self$n_num_features = n_num_features
    self$n_cat_features = length(cat_cardinalities)
    self$arch_type = arch_type
    self$num_module = num_embeddings
    self$cat_module = if (length(cat_cardinalities)) nn_tabm_one_hot(cat_cardinalities) else NULL

    # Upstream `TabM.make()`: "normal" if there are non-trivial modules before the
    # backbone (i.e. num_embeddings), "random-signs" otherwise.
    start_scaling_init = start_scaling_init %??%
      if (is.null(num_embeddings)) "random-signs" else "normal"
    # Upstream `TabM.make()`: 2 blocks with embeddings, 3 without.
    n_blocks = n_blocks %??% if (is.null(num_embeddings)) 3L else 2L

    self$k = k
    self$ensemble_view = nn_tabm_ensemble_view(k = k)
    self$backbone = tabm_make_backbone(
      d_in = sum(d_features), n_blocks = n_blocks, d_block = d_block, dropout = dropout,
      activation = activation, k = k, arch_type = arch_type,
      start_scaling_init = start_scaling_init,
      start_scaling_init_chunks = if (is.null(start_scaling_init)) NULL else d_features
    )
    self$output = if (is.null(d_out)) {
      NULL
    } else {
      nn_tabm_linear_ensemble(self$backbone$d_out, d_out, k = k)
    }
  },
  forward = function(x_num = NULL, x_cat = NULL) {
    # When a task has only one input tensor, mlr3torch calls the network *by position*
    # (see `learner_torch_train()`), so a purely categorical task arrives in `x_num`.
    if (self$n_num_features == 0L && is.null(x_cat)) {
      x_cat = x_num
      x_num = NULL
    }
    if (self$n_num_features > 0L && is.null(x_num)) {
      stopf("nn_tabm was built with %i numerical features, but x_num is NULL.", self$n_num_features)
    }
    if (self$n_cat_features > 0L && is.null(x_cat)) {
      stopf("nn_tabm was built with %i categorical features, but x_cat is NULL.", self$n_cat_features)
    }
    parts = list()
    if (!is.null(x_num)) {
      x_n = if (is.null(self$num_module)) x_num else self$num_module(x_num)
      # (B, n_num, d_embedding) -> (B, n_num * d_embedding); a no-op without embeddings
      parts[[length(parts) + 1L]] = x_n$flatten(start_dim = 2L)
    }
    if (!is.null(x_cat)) parts[[length(parts) + 1L]] = self$cat_module(x_cat)
    x = if (length(parts) == 1L) parts[[1L]] else torch_cat(parts, dim = 2L)

    x = self$ensemble_view(x)
    x = self$backbone(x)
    if (!is.null(self$output)) x = self$output(x)
    x
  }
)

# --------------------------------------------------------------------------------------
# Loss adapter
# --------------------------------------------------------------------------------------

# Ported from `loss_fn()` in `paper/bin/model.py`.
#' @title Ensemble Loss Adapter for TabM
#'
#' @description
#' Adapts a loss function to the `(batch, k, d_out)` output of [`nn_tabm()`]:
#' the ensemble dimension is folded into the batch dimension and every target is repeated
#' `k` times, so that all `k` submodels are trained on the full batch.
#' This is what the reference implementation of TabM does.
#'
#' [`LearnerTorchTabM`] applies this adapter automatically to whatever loss it is
#' configured with; it only needs to be used explicitly when combining [`nn_tabm()`] with
#' [`LearnerTorchModule`].
#'
#' @param loss ([`nn_module`][torch::nn_module])\cr
#'   The loss module that is applied to the folded prediction and the repeated target.
#'
#' @references
#' `r format_bib("gorishniy2025tabm")`
#'
#' @noRd
#' @examplesIf torch::torch_is_installed()
#' loss = nn_tabm_loss(torch::nn_cross_entropy_loss())
#' input = torch::torch_randn(6, 4, 3)
#' target = torch::torch_randint(1, 3, 6, dtype = torch::torch_long())
#' loss(input, target)
nn_tabm_loss = nn_module("nn_tabm_loss",
  initialize = function(loss) {
    self$loss = loss
  },
  forward = function(input, target) {
    k = input$shape[2L]
    self$loss(
      input$flatten(start_dim = 1L, end_dim = 2L),
      target$repeat_interleave(k, dim = 1L)
    )
  }
)


# --------------------------------------------------------------------------------------
# The learner
# --------------------------------------------------------------------------------------

# Build the `num_embeddings` module from the learner's parameter values.
# The defaults follow the official TabM usage example
# (https://github.com/yandex-research/tabm/blob/main/example.ipynb):
#   LinearReLUEmbeddings(n)                                   -> d_embedding = 32 (upstream default)
#   PeriodicEmbeddings(n, lite = FALSE)                       -> d_embedding = 24 (upstream default)
#   PiecewiseLinearEmbeddings(bins, 16, activation = FALSE)
tabm_make_num_embeddings = function(type, n_num_features, param_vals, x_num = NULL) {
  if (is.null(type) || identical(type, "none")) {
    return(NULL)
  }
  if (n_num_features == 0L) {
    stopf("The parameter 'num_embeddings' is set to '%s', but the task has no numerical features.", type)
  }
  switch(type,
    "linear_relu" = nn_linear_relu_embeddings(
      n_num_features,
      d_embedding = param_vals$d_embedding %??% 32L
    ),
    "periodic" = nn_periodic_embeddings(
      n_num_features,
      d_embedding = param_vals$d_embedding %??% 24L,
      n_frequencies = param_vals$n_frequencies %??% 48L,
      frequency_init_scale = param_vals$frequency_init_scale %??% 0.01,
      activation = param_vals$embedding_activation %??% TRUE,
      lite = param_vals$lite %??% FALSE
    ),
    "piecewise_linear" = {
      n_bins = param_vals$n_bins %??% 48L
      bins = tryCatch(compute_bins(x_num, n_bins = n_bins), error = function(e) {
        stopf("Cannot compute the bins for the piecewise-linear embeddings (n_bins = %i): %s", n_bins, conditionMessage(e)) # nolint
      })
      nn_piecewise_linear_embeddings(
        bins,
        d_embedding = param_vals$d_embedding %??% 16L,
        activation = param_vals$embedding_activation %??% FALSE
      )
    },
    stopf("Unknown num_embeddings type '%s'.", type)
  )
}

#' @title TabM
#'
#' @templateVar name tabm
#' @templateVar task_types classif, regr
#' @templateVar param_vals k = 4, n_blocks = 2, d_block = 32
#' @template params_learner
#' @template learner
#' @template learner_example
#'
#' @description
#' TabM is an MLP-based tabular deep learning model that efficiently represents an
#' ensemble of `k` MLPs: the `k` submodels are trained in parallel on the same batches
#' and share most of their weights, which acts as a strong regularizer.
#' The network produces `k` predictions per observation; the learner averages the
#' predicted *probabilities* (classification) or the predicted values (regression)
#' over the `k` submodels, and its loss function trains all `k` submodels jointly.
#'
#' Numerical features are used as-is, or -- if the `num_embeddings` parameter is set --
#' embedded feature-wise first, which usually improves the performance considerably.
#' Categorical features are one-hot encoded.
#'
#' @section Parameters:
#' Parameters from [`LearnerTorch`], as well as:
#' * `arch_type` :: `character(1)`\cr
#'   The architecture type, one of:
#'   * `"tabm"` (default) -- BatchEnsemble with the TabM initialization, i.e. all
#'     multiplicative adapters except the very first one are initialized with ones.
#'   * `"tabm-mini"` -- all non-shared parameters are concentrated in a single
#'     elementwise affine transformation applied to the input.
#' * `k` :: `integer(1)`\cr
#'   The number of ensemble members. Default is `32`.
#' * `n_blocks` :: `integer(1)`\cr
#'   The number of blocks of the MLP backbone.
#'   If unset, `2` is used when `num_embeddings` is set and `3` otherwise.
#' * `d_block` :: `integer(1)`\cr
#'   The width of the MLP backbone. Default is `512`.
#' * `dropout` :: `numeric(1)`\cr
#'   The dropout rate. Default is `0.1`.
#' * `activation` :: `character(1)`, `nn_module_generator` or `function`\cr
#'   The activation function of the MLP backbone. Either the name of an activation of
#'   the `torch` package (e.g. `"relu"`, `"nn_relu"` or `"ReLU"`), an
#'   [`nn_module_generator`][torch::nn_module] such as [`nn_relu`][torch::nn_relu], or a
#'   function returning an [`nn_module`][torch::nn_module]. Default is `"relu"`.
#' * `start_scaling_init` :: `character(1)`\cr
#'   The initialization of the very first (non-shared) scaling, either `"random-signs"`
#'   or `"normal"`. If unset, `"normal"` is
#'   used when `num_embeddings` is set and `"random-signs"` otherwise.
#'
#' Parameters of the embeddings for the numerical features:
#' * `num_embeddings` :: `character(1)`\cr
#'   The type of the numerical feature embeddings, one of `"none"` (default),
#'   `"linear_relu"`, `"periodic"` or `"piecewise_linear"`. The last two usually perform best.
#' * `d_embedding` :: `integer(1)`\cr
#'   The embedding size. If unset, `32` is used for `"linear_relu"`, `24` for
#'   `"periodic"` and `16` for `"piecewise_linear"`.
#' * `n_frequencies` :: `integer(1)`\cr
#'   `"periodic"` only: the number of frequencies per feature. Default is `48`.
#' * `frequency_init_scale` :: `numeric(1)`\cr
#'   `"periodic"` only: the initialization scale of the frequencies. This is an
#'   important hyperparameter. Default is `0.01`.
#' * `lite` :: `logical(1)`\cr
#'   `"periodic"` only: whether the outer linear layer is shared between all features.
#'   Default is `FALSE`.
#' * `embedding_activation` :: `logical(1)`\cr
#'   `"periodic"` and `"piecewise_linear"` only: whether a ReLU is applied at the end of
#'   the embedding. If unset, `TRUE` is used for `"periodic"` and `FALSE` for
#'   `"piecewise_linear"`.
#' * `n_bins` :: `integer(1)`\cr
#'   `"piecewise_linear"` only: the number of quantile bins, computed from the training
#'   data. Must be smaller than the number of training observations. Default is `48`.
#'
#' @section Loss and Prediction:
#' The network output has shape `(batch, k, d_out)`.
#' At training time the learner therefore applies the configured loss to the `k` predictions
#' separately: the ensemble dimension is folded into the batch dimension and each target is
#' repeated `k` times. `$loss` itself is left untouched and stays whatever was configured.
#' For prediction, the per-submodel probabilities (softmax for multiclass, sigmoid for
#' binary) are averaged over the `k` submodels; for regression the outputs are averaged.
#'
#' @references
#' `r format_bib("gorishniy2025tabm", "wen2020batchensemble")`
#' @export
LearnerTorchTabM = R6Class("LearnerTorchTabM",
  inherit = LearnerTorch,
  public = list(
    #' @description
    #' Creates a new instance of this [R6][R6::R6Class] class.
    initialize = function(task_type, optimizer = NULL, loss = NULL, callbacks = list()) {
      check_activation = crate(function(x) {
        if (test_string(x, min.chars = 1L) || is.function(x)) {
          return(TRUE)
        }
        "must be a character(1), an nn_module_generator, or a function returning an nn_module"
      })

      private$.param_set_base = ps(
        arch_type = p_fct(levels = c("tabm", "tabm-mini"),
          init = "tabm", tags = "train"),
        k = p_int(lower = 1L, init = 32L, tags = "train"),
        # no init: the default depends on `num_embeddings` (upstream `TabM.make()`)
        n_blocks = p_int(lower = 1L, tags = "train"),
        d_block = p_int(lower = 1L, init = 512L, tags = "train"),
        dropout = p_dbl(lower = 0, upper = 1, init = 0.1, tags = "train"),
        activation = p_uty(init = "relu", tags = "train", custom_check = check_activation),
        # no init: the default depends on `num_embeddings` (upstream `TabM.make()`)
        start_scaling_init = p_fct(levels = c("random-signs", "normal"), tags = "train"),
        num_embeddings = p_fct(
          levels = c("none", "linear_relu", "periodic", "piecewise_linear"),
          init = "none", tags = "train"),
        d_embedding = p_int(lower = 1L, tags = "train"),
        n_frequencies = p_int(lower = 1L, init = 48L, tags = "train"),
        # `nn_periodic()` requires a strictly positive scale, but that cannot be expressed
        # here: paradox accepts values within `sqrt(.Machine$double.eps)` of a bound, so any
        # bound small enough to be honest is ignored, and `tolerance = 0` cannot be combined
        # with `init`. A scale of 0 is therefore only rejected when the network is built.
        frequency_init_scale = p_dbl(lower = 0, init = 0.01, tags = "train"),
        lite = p_lgl(init = FALSE, tags = "train"),
        embedding_activation = p_lgl(tags = "train"),
        n_bins = p_int(lower = 2L, init = 48L, tags = "train")
      )

      super$initialize(
        task_type = task_type,
        id = paste0(task_type, ".tabm"),
        label = "TabM",
        param_set = alist(private$.param_set_base),
        optimizer = optimizer,
        callbacks = callbacks,
        loss = loss,
        man = "mlr3torch::mlr_learners.tabm",
        feature_types = c("numeric", "integer", "logical", "factor", "ordered"),
        jittable = TRUE
      )
    }
  ),
  private = list(
    # The network returns one prediction per ensemble member, so the configured loss is
    # applied to the `k` predictions separately, see the Loss and Prediction section.
    .loss_fn = function(task, param_vals) {
      nn_tabm_loss(super$.loss_fn(task, param_vals))
    },
    .ingress_tokens = function(task, param_vals) {
      n_num = n_num_features(task)
      n_categ = n_categ_features(task)
      if (n_num == 0L && n_categ == 0L) {
        stopf("Learner '%s' received task '%s' without any supported features.", self$id, task$id)
      }
      out = list()
      if (n_num > 0L) {
        out$x_num = ingress_num(shape = c(NA, n_num))
      }
      if (n_categ > 0L) {
        out$x_cat = ingress_categ(shape = c(NA, n_categ))
      }
      out
    },
    .network = function(task, param_vals) {
      arch_type = param_vals$arch_type %??% "tabm"
      n_num = n_num_features(task)

      # the bins of the piecewise-linear embeddings must be computed on the training data
      x_num = if (identical(param_vals$num_embeddings, "piecewise_linear") && n_num > 0L) {
        # the ingress token defines the column order the network will see, and it is not
        # always the order of `task$feature_names` (e.g. after `po("scale")`)
        num_features = ingress_num()$features(task)
        batchgetter_num(task$data(cols = num_features))
      }
      num_embeddings = tabm_make_num_embeddings(param_vals$num_embeddings, n_num, param_vals, x_num)

      nn_tabm(
        n_num_features = n_num,
        cat_cardinalities = unname(categ_cardinalities(task)),
        d_out = output_dim_for(task),
        num_embeddings = num_embeddings,
        arch_type = arch_type,
        k = param_vals$k %??% 32L,
        # NULL lets nn_tabm() apply upstream's `TabM.make()` defaults, which depend on
        # whether embeddings are used
        n_blocks = param_vals$n_blocks,
        d_block = param_vals$d_block %??% 512L,
        dropout = param_vals$dropout %??% 0.1,
        activation = param_vals$activation %??% "relu",
        start_scaling_init = param_vals$start_scaling_init
      )
    },
    # The network returns one prediction per submodel, i.e. a tensor of shape
    # (batch, k, d_out). Upstream averages the probabilities of the k submodels
    # (see `paper/bin/model.py`), which is not the same as averaging the logits.
    # `encode_prediction_default()` expects scores, so the averaged probabilities are
    # mapped back to the score scale (log / logit); this roundtrip is exact up to
    # floating point accuracy because softmax(log(p)) == p and sigmoid(logit(p)) == p.
    .encode_prediction = function(predict_tensor, task) {
      reduced = with_no_grad({
        if (task$task_type == "regr") {
          predict_tensor$mean(dim = 2L)
        } else if ("twoclass" %in% task$properties) {
          p = torch_sigmoid(predict_tensor)$mean(dim = 2L)$clamp(min = 1e-7, max = 1 - 1e-7)
          torch_log(p) - torch_log1p(-p)
        } else {
          p = nnf_softmax(predict_tensor, dim = 3L)$mean(dim = 2L)
          torch_log(p$clamp(min = 1e-30))
        }
      })
      encode_prediction_default(
        predict_tensor = reduced,
        predict_type = self$predict_type,
        task = task
      )
    }
  )
)

#' @include aaa.R
register_learner("classif.tabm", LearnerTorchTabM)
register_learner("regr.tabm", LearnerTorchTabM)
