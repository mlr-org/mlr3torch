skip_if(running_on_mac())
# Threads must be 1 on macOS per ?torch_set_num_threads

test_that("with_torch_settings leaves global state (mostly) untouched", {
  # Function to reset the torch manual seed cannot yet be used (wait for release after 0.10.0)
  runif(1)

  prev_num_threads = 10
  torch_set_num_threads(prev_num_threads)
  prev_seed = get0(".Random.seed", globalenv(), mode = "integer", inherits = FALSE)
  prev_torch_rng_state = torch_get_rng_state()

  with_torch_settings(1, 1, {
    x1 = runif(1)
    y1 = torch_randn(1)
  })

  with_torch_settings(1, 1, {
    x2 = runif(1)
    y2 = torch_randn(1)
  })

  # Results are reproducible
  expect_true(x1 == x2)
  expect_true(torch_equal(y1, y2))
  expect_true(torch_equal(prev_torch_rng_state, torch_get_rng_state()))

  expect_equal(torch_get_num_threads(), prev_num_threads)
  expect_equal(get0(".Random.seed", globalenv(), mode = "integer", inherits = FALSE), prev_seed)


  # We have checked that within with_torch_settings() everything is as expected,
  # Now we check that not everything afterwards is deterministic
  # (This would happen if we did not set the seed afterwards back to the previous value)

  withr::with_seed(10, {
    with_torch_settings(seed = 1, num_threads = 1, NULL)
    a = runif(1)
    at = torch_randn(1)
  })

  withr::with_seed(20, {
    with_torch_settings(seed = 1, num_threads = 1, NULL)
    b = runif(1)
    bt = torch_randn(1)
  })
  expect_false(all(a == b))
  expect_false(torch_equal(at, bt))
})
