test_that("with_torch_settings leaves global state untouched", {
  runif(1)

  if (!running_on_mac()) {
    prev_num_threads = 10
    torch_set_num_threads(prev_num_threads)
  } else {
    prev_num_threads = torch_get_num_threads()
  }
  prev_torch_rng_state = torch_get_rng_state()

  with_torch_settings(1, 1, 1, {
    y1 = torch_randn(1)
  })

  with_torch_settings(1, 1, 1, {
    y2 = torch_randn(1)
  })

  # Results are reproducible
  expect_true(torch_equal(y1, y2))
  expect_true(torch_equal(prev_torch_rng_state, torch_get_rng_state()))

  expect_equal(torch_get_num_threads(), prev_num_threads)

  # We have checked that within with_torch_settings() everything is as expected,
  # Now we check that not everything afterwards is deterministic
  # (This would happen if we did not set the seed afterwards back to the previous value)

  withr::with_seed(10, {
    with_torch_settings(seed = 1, num_threads = 1, num_interop_threads = 1, NULL)
    at = torch_randn(1)
  })

  withr::with_seed(20, {
    with_torch_settings(seed = 1, num_threads = 1, num_interop_threads = 1, NULL)
    bt = torch_randn(1)
  })
  expect_false(torch_equal(at, bt))
})

test_that("num_interop_threads is only set when the user asks for it", {
  skip_if_not_installed("callr")
  skip_on_cran()
  # The interop count can be set only once per session and is never restored, so the behaviour only
  # exists in a session that has not trained anything yet -- hence the subprocesses.
  mlp = function(...) lrn("classif.mlp", epochs = 1, batch_size = 50, neurons = 5, device = "cpu", ...)

  # the default must not consume the one chance to set it
  default = callr::r(function(mlp) {
    library(mlr3torch)
    before = torch::torch_get_num_interop_threads()
    mlp()$train(tsk("iris"))
    c(before, torch::torch_get_num_interop_threads())
  }, args = list(mlp = mlp))
  expect_equal(default[1L], default[2L])

  # and because it does not, a learner carrying an explicit value still takes effect afterwards
  explicit = callr::r(function(mlp) {
    library(mlr3torch)
    # a value that differs from whatever this machine defaults to
    target = if (torch::torch_get_num_interop_threads() == 2L) 3L else 2L
    mlp()$train(tsk("iris"))
    mlp(num_interop_threads = target)$train(tsk("iris"))
    c(target, torch::torch_get_num_interop_threads())
  }, args = list(mlp = mlp))
  expect_equal(explicit[1L], explicit[2L])
})

test_that("interop threads proper warning message", {
  skip_if_not_installed("callr")
  # otherwise capture.output does for some reason not capture the warning message 
  skip_if(!running_on_mac())

  result = callr::r(function() {
    library(torch)
    with_torch_settings = getFromNamespace("with_torch_settings", "mlr3torch")
    with_torch_settings(NULL, 1, 2, invisible(NULL))
    x1 = capture.output(with_torch_settings(NULL, 1, 2, invisible(NULL)))
    x2 = capture.output(with_torch_settings(NULL, 1, 1, invisible(NULL)))
    list(x1, x2)
  })
  expect_true(length(result[[1]]) == 0)
  expect_true(grepl("keeping the previous value 2", result[[2]], fixed = TRUE))
})
