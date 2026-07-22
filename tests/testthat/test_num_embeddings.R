test_that("nn_linear_embeddings works", {
  m = nn_linear_embeddings(3L, 4L)
  expect_class(m, "nn_linear_embeddings")
  expect_equal(m$get_output_shape(), c(3L, 4L))
  expect_equal(m(torch_randn(5, 3))$shape, c(5, 3, 4))
  # each feature is embedded by its own affine map
  x = torch_randn(2, 3)
  manual = x$unsqueeze(-1L) * m$weight + m$bias
  expect_true(as.logical((torch_abs(m(x) - manual) < 1e-6)$all()))
  expect_error(m(torch_randn(5, 2)), "last dimension of the input")
})

test_that("nn_linear_relu_embeddings works", {
  m = nn_linear_relu_embeddings(3L, 8L)
  expect_equal(m$get_output_shape(), c(3L, 8L))
  out = m(torch_randn(5, 3))
  expect_equal(out$shape, c(5, 3, 8))
  expect_true(as.logical((out >= 0)$all()))
  # the upstream default embedding size
  expect_equal(nn_linear_relu_embeddings(3L)$get_output_shape(), c(3L, 32L))
})

test_that("nn_periodic_embeddings works", {
  m = nn_periodic_embeddings(3L, 8L, n_frequencies = 5L, lite = FALSE)
  expect_equal(m$get_output_shape(), c(3L, 8L))
  expect_equal(m(torch_randn(5, 3))$shape, c(5, 3, 8))
  # without `lite`, every feature has its own outer linear layer
  expect_class(m$linear, "nn_nlinear")
  expect_equal(m$linear$weight$shape, c(3, 10, 8))

  m = nn_periodic_embeddings(3L, 8L, n_frequencies = 5L, lite = TRUE)
  expect_class(m$linear, "nn_linear")
  expect_equal(m(torch_randn(5, 3))$shape, c(5, 3, 8))

  torch_manual_seed(1)
  m = nn_periodic_embeddings(3L, 8L, n_frequencies = 5L, activation = FALSE, lite = FALSE)
  expect_null(m$activation)
  expect_true(as.logical((m(torch_randn(50, 3)) < 0)$any()))

  expect_error(nn_periodic_embeddings(3L, 8L, activation = FALSE, lite = TRUE),
    "lite = TRUE is allowed only when activation = TRUE")
  # the upstream defaults
  expect_equal(nn_periodic_embeddings(3L, lite = FALSE)$get_output_shape(), c(3L, 24L))
  expect_equal(nn_periodic_embeddings(3L, lite = FALSE)$periodic$weight$shape, c(3, 48))
})

test_that("the frequencies are initialized from a truncated normal", {
  torch_manual_seed(1)
  sigma = 0.05
  m = nn_periodic_embeddings(500L, 4L, n_frequencies = 200L, frequency_init_scale = sigma,
    lite = TRUE)
  w = m$periodic$weight$detach()
  expect_true(as.logical((torch_abs(w) <= 3 * sigma)$all()))
  # the sd of N(0, sigma) truncated at +- 3 sigma
  expected_sd = sigma * sqrt(1 - 2 * 3 * dnorm(3) / (2 * pnorm(3) - 1))
  expect_equal(as.numeric(w$std()), expected_sd, tolerance = 0.02)
  expect_equal(as.numeric(w$mean()), 0, tolerance = 0.01)
})

test_that("compute_bins works", {
  set.seed(1)
  x = matrix(rnorm(1000), ncol = 5L)
  bins = compute_bins(x, n_bins = 10L)
  expect_list(bins, types = "torch_tensor", len = 5L)
  for (b in bins) {
    expect_equal(b$dim(), 1L)
    expect_true(b$shape[1L] <= 11L)
    # sorted and within the data range
    expect_true(as.logical((torch_diff(b) > 0)$all()))
  }
  # the outer bin edges are the minimum and the maximum
  expect_equal(as.numeric(bins[[1L]][1L]), min(x[, 1L]), tolerance = 1e-6)
  expect_equal(as.numeric(bins[[1L]][bins[[1L]]$shape[1L]]), max(x[, 1L]), tolerance = 1e-6)

  # coinciding quantiles are removed, so a feature can end up with fewer bins
  x = cbind(rnorm(100), rep(c(0, 1), 50L))
  bins = compute_bins(x, n_bins = 10L)
  expect_equal(bins[[1L]]$shape[1L], 11L)
  # only the edges 0, 0.5 and 1 remain for the binary column
  expect_equal(bins[[2L]]$shape[1L], 3L)
  expect_equal(as.numeric(bins[[2L]]), c(0, 0.5, 1), tolerance = 1e-6)

  expect_error(compute_bins(matrix(1, nrow = 10L, ncol = 2L), n_bins = 5L),
    "at least two distinct values")
  expect_error(compute_bins(matrix(rnorm(20), ncol = 2L), n_bins = 20L),
    "smaller than the number of rows")
  expect_error(compute_bins(matrix(c(NA_real_, rnorm(19)), ncol = 2L), n_bins = 5L),
    "must not contain NaN")
})

test_that("nn_piecewise_linear_embeddings works", {
  bins = compute_bins(matrix(rnorm(1000), ncol = 5L), n_bins = 10L)
  m = nn_piecewise_linear_embeddings(bins, d_embedding = 8L, activation = FALSE,
    version = "B")
  expect_equal(m$get_output_shape(), c(5L, 8L))
  expect_equal(m(torch_randn(6, 5))$shape, c(6, 5, 8))
  # version "B" adds the linear shortcut and zero-initializes the piecewise-linear part,
  # so the module starts out as a plain linear embedding
  expect_class(m$linear0, "nn_linear_embeddings")
  expect_true(as.logical((m$linear$weight == 0)$all()))
  x = torch_randn(4, 5)
  expect_true(as.logical((torch_abs(m(x) - m$linear0(x)) < 1e-6)$all()))

  m = nn_piecewise_linear_embeddings(bins, d_embedding = 8L, activation = TRUE,
    version = "A")
  expect_null(m$linear0)
  out = m(torch_randn(6, 5))
  expect_equal(out$shape, c(6, 5, 8))
  expect_true(as.logical((out >= 0)$all()))

  expect_error(nn_piecewise_linear_embeddings(bins, 8L, activation = TRUE, version = "C"),
    "version")
  expect_error(nn_piecewise_linear_embeddings(bins, 8L, activation = TRUE),
    "argument \"version\" is missing")
  expect_error(m(torch_randn(2, 6, 5)), "exactly one batch dimension")
})

test_that("the piecewise-linear encoding is correct", {
  # one feature with the bin edges 0, 1, 3 (i.e. two bins of width 1 and 2)
  bins = list(torch_tensor(c(0, 1, 3)))
  impl = nn_piecewise_linear_encoding_impl(bins)
  expect_equal(impl$get_max_n_bins(), 2L)
  x = torch_tensor(matrix(c(-1, 0, 0.5, 1, 2, 3, 4), ncol = 1L))
  out = as.matrix(impl(x)$squeeze(2L))
  # first component: min(x / 1, 1); second: max((x - 1) / 2, 0)
  expect_equal(out[, 1L], pmin(as.numeric(x), 1), tolerance = 1e-6)
  expect_equal(out[, 2L], pmax((as.numeric(x) - 1) / 2, 0), tolerance = 1e-6)

  # a feature with a single bin behaves like min-max scaling (no lower clamp)
  bins = list(torch_tensor(c(0, 1, 3)), torch_tensor(c(0, 2)))
  impl = nn_piecewise_linear_encoding_impl(bins)
  expect_false(is.null(impl$single_bin_mask))
  out = impl(torch_tensor(matrix(c(0.5, -1), nrow = 1L)))
  expect_equal(as.numeric(out[1, 2, 2]), -0.5, tolerance = 1e-6)
})

test_that("all embedding modules can be jit traced", {
  bins = compute_bins(matrix(rnorm(400), ncol = 4L), n_bins = 5L)
  modules = list(
    nn_linear_embeddings(4L, 6L),
    nn_linear_relu_embeddings(4L, 6L),
    nn_periodic_embeddings(4L, 6L, n_frequencies = 5L, lite = FALSE),
    nn_periodic_embeddings(4L, 6L, n_frequencies = 5L, lite = TRUE),
    nn_piecewise_linear_embeddings(bins, 6L, activation = FALSE, version = "B")
  )
  for (m in modules) {
    m$eval()
    traced = jit_trace(m, torch_randn(5, 4))
    # a different batch size must still work
    expect_equal(traced(torch_randn(3, 4))$shape, c(3, 4, 6))
    expect_true(as.logical((torch_abs(traced(torch_randn(3, 4)) - m(torch_randn(3, 4))) < 1e6)$all()))
  }
})

test_that("gradients reach every parameter of the embedding modules", {
  bins = compute_bins(matrix(rnorm(400), ncol = 4L), n_bins = 5L)
  modules = list(
    nn_linear_embeddings(4L, 6L),
    nn_linear_relu_embeddings(4L, 6L),
    nn_periodic_embeddings(4L, 6L, n_frequencies = 5L, lite = FALSE),
    nn_piecewise_linear_embeddings(bins, 6L, activation = FALSE, version = "A")
  )
  for (m in modules) {
    m(torch_randn(8, 4))$sum()$backward()
    for (p in m$parameters) {
      expect_false(is_undefined_tensor(p$grad))
      expect_true(as.logical((p$grad != 0)$any()))
    }
  }
})
