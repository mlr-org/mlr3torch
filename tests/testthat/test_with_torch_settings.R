test_that("with_torch_settings leaves global state untouched", {
  runif(1)

  if (!running_on_mac()) {
    prev_num_threads = 10
    torch_set_num_threads(prev_num_threads)
  } else {
    prev_num_threads = torch_get_num_threads()
  }
  prev_torch_rng_state = torch_get_rng_state()

  with_torch_settings(1, 1, {
    y1 = torch_randn(1)
  })

  with_torch_settings(1, 1, {
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
    with_torch_settings(seed = 1, num_threads = 1, NULL)
    at = torch_randn(1)
  })

  withr::with_seed(20, {
    with_torch_settings(seed = 1, num_threads = 1, NULL)
    bt = torch_randn(1)
  })
  expect_false(torch_equal(at, bt))
})
