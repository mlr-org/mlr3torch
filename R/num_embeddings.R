# ======================================================================================
# Embeddings for numerical features
# ("On Embeddings for Numerical Features in Tabular Deep Learning", NeurIPS 2022)
# ======================================================================================
#
# Upstream repository : https://github.com/yandex-research/rtdl-num-embeddings
# Ported from         : the single-file package `rtdl_num_embeddings.py`, version 0.0.12,
#                       commit a8fc25025c83f2321c63ff127a3bcef83bb1bfb5
#                       (the installed wheel `rtdl_num_embeddings==0.0.12` is
#                       byte-identical to `main`).
# Upstream license    : MIT
#                       (https://github.com/yandex-research/rtdl-num-embeddings/blob/main/LICENSE)
#
# Ported components (upstream name -> name here):
#   LinearEmbeddings              -> nn_linear_embeddings
#   LinearReLUEmbeddings          -> nn_linear_relu_embeddings
#   _Periodic                     -> nn_periodic
#   _NLinear                      -> nn_nlinear
#   PeriodicEmbeddings            -> nn_periodic_embeddings
#   _PiecewiseLinearEncodingImpl  -> nn_piecewise_linear_encoding_impl
#   PiecewiseLinearEmbeddings     -> nn_piecewise_linear_embeddings
#   compute_bins                  -> compute_bins
#
# Intentional deviations from upstream:
#
#  * `compute_bins()` implements the **quantile-based** binning only (Section 3.2.1 of
#    the paper). The tree-based binning (Section 3.2.2) requires
#    `sklearn.tree.DecisionTree{Regressor,Classifier}` and is therefore not portable
#    without adding a new package dependency, which is not allowed here. The
#    `tree_kwargs` / `y` / `regression` / `verbose` arguments do not exist.
#  * The standalone `PiecewiseLinearEncoding` module is not ported: it is not used by
#    TabM (which uses `PiecewiseLinearEmbeddings`), and its masked-flatten forward pass
#    relies on boolean advanced indexing that has no clean R torch equivalent. The
#    encoding itself (`_PiecewiseLinearEncodingImpl`) *is* ported, since
#    `PiecewiseLinearEmbeddings` needs it.
#  * `nn_piecewise_linear_embeddings()` requires `version` to be given explicitly.
#    Upstream falls back to `version = "A"` with a deprecation warning and states that
#    "In future, omitting this argument will result in an exception".
#  * `compute_bins()` additionally accepts a `matrix` / `data.frame` and reports the
#    offending column *indices* in its error messages.
#  * The frequencies of `nn_periodic_embeddings()` are drawn with R torch's
#    `nn_init_trunc_normal_()`, which implements a different sampling algorithm than
#    PyTorch's `nn.init.trunc_normal_()`. The *distribution* is the same (a normal with
#    mean 0 and standard deviation `frequency_init_scale`, truncated at +/- 3 standard
#    deviations), but the two do not consume the RNG stream identically, so a given
#    manual seed does not reproduce the same numbers in both languages. All other
#    modules of this file do reproduce PyTorch's initialization exactly.
#  * `nn_piecewise_linear_encoding_impl()` does not register the `mask` buffer of
#    upstream's `_PiecewiseLinearEncodingImpl`; it is only read by the unported
#    `PiecewiseLinearEncoding` wrapper, never in the forward pass.
#
# ======================================================================================

# Upstream: `_check_input_shape()`.
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

#' @title Linear Embeddings for Numerical Features
#' @name nn_linear_embeddings
#'
#' @description
#' Embeds each numerical feature with its own (scalar) linear transformation, i.e.
#' feature `i` is mapped to `x[i] * weight[i, ] + bias[i, ]`.
#' For an input of shape `(*, n_features)` the output shape is
#' `(*, n_features, d_embedding)`.
#'
#' @param n_features (`integer(1)`)\cr
#'   The number of numerical features.
#' @param d_embedding (`integer(1)`)\cr
#'   The embedding size.
#'
#' @references
#' `r format_bib("gorishniy2022embeddings")`
#'
#' @export
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

#' @title Linear-ReLU Embeddings for Numerical Features
#' @name nn_linear_relu_embeddings
#'
#' @description
#' [`nn_linear_embeddings()`] followed by a ReLU activation.
#' For an input of shape `(*, n_features)` the output shape is
#' `(*, n_features, d_embedding)`.
#'
#' @param n_features (`integer(1)`)\cr
#'   The number of numerical features.
#' @param d_embedding (`integer(1)`)\cr
#'   The embedding size. Default is `32`.
#'
#' @references
#' `r format_bib("gorishniy2022embeddings")`
#'
#' @export
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

# Upstream: `_Periodic`. Must not be used directly.
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
    # NOTE[DIFF] (upstream): extreme values (~0.3% probability) are explicitly avoided.
    bound = self$sigma * 3
    nn_init_trunc_normal_(self$weight, mean = 0, std = self$sigma, a = -bound, b = bound)
  },
  forward = function(input) {
    check_num_embeddings_input(input, self$n_features)
    x = 2 * pi * self$weight * input$unsqueeze(-1L)
    torch_cat(list(torch_cos(x), torch_sin(x)), dim = -1L)
  }
)

# Upstream: `_NLinear`, i.e. n separate linear layers, one per feature embedding.
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
#' @export
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
      # NOTE[DIFF] (upstream): the lite variation was introduced by the TabR paper.
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

# Upstream: `_check_bins()`.
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
#' @export
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

  # NOTE (upstream): removing identical quantiles *after* computing them is not the same
  # as limiting the number of quantiles by the number of distinct values.
  quantiles = torch_quantile(x, torch_linspace(0, 1, n_bins + 1L)$to(dtype = x$dtype), dim = 1L)
  bins = lapply(seq_len(p), function(j) {
    # the quantiles are sorted, so consecutive uniqueness is the same as uniqueness
    torch_unique_consecutive(quantiles[, j])[[1L]]
  })
  check_bins(bins)
  bins
}

# Upstream: `_PiecewiseLinearEncodingImpl`. Must not be used directly (it adds no
# positional information to the feature encodings).
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
#' @param version (`character(1)`)\cr
#'   Either `"A"` (the version of the original paper) or `"B"` (introduced by the TabM
#'   paper; adds a [`nn_linear_embeddings()`] shortcut and zero-initializes the
#'   piecewise-linear part, so that the module behaves like a linear embedding at
#'   initialization). `"B"` is required by [`nn_tabm()`].
#'   Unlike upstream, this argument must be given explicitly.
#'
#' @references
#' `r format_bib("gorishniy2022embeddings", "gorishniy2025tabm")`
#'
#' @export
#' @examplesIf torch::torch_is_installed()
#' bins = compute_bins(matrix(rnorm(200), ncol = 2), n_bins = 4)
#' m = nn_piecewise_linear_embeddings(bins, d_embedding = 8, activation = FALSE,
#'   version = "B")
#' m(torch::torch_randn(3, 2))$shape
nn_piecewise_linear_embeddings = nn_module("nn_piecewise_linear_embeddings",
  initialize = function(bins, d_embedding, activation, version) {
    d_embedding = assert_int(d_embedding, lower = 1L, coerce = TRUE)
    assert_flag(activation)
    version = assert_choice(version, c("A", "B"))
    check_bins(bins)

    n_features = length(bins)
    # NOTE[DIFF] (upstream): version "B" was introduced by the TabM paper.
    is_version_b = version == "B"

    self$linear0 = if (is_version_b) nn_linear_embeddings(n_features, d_embedding) else NULL
    self$impl = nn_piecewise_linear_encoding_impl(bins)
    self$linear = nn_nlinear(n_features, self$impl$get_max_n_bins(), d_embedding,
      # for version "B" the bias is already part of linear0
      bias = !is_version_b)
    if (is_version_b) {
      # Because of this, the whole embedding behaves like a linear embedding at
      # initialization; the piecewise-linear component is learnt incrementally.
      nn_init_zeros_(self$linear$weight)
    }
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
    x_linear = if (is.null(self$linear0)) NULL else self$linear0(input)
    x_ple = self$impl(input)
    x_ple = self$linear(x_ple)
    if (!is.null(self$activation)) x_ple = self$activation(x_ple)
    if (is.null(x_linear)) x_ple else x_linear + x_ple
  }
)
