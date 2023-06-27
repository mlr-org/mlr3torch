test_that("auto paramtest works", {
  # captures missing parameters
  f1 = function(x, y) NULL
  ps1 = ps(x = p_uty())
  res1 = autotest_paramset(ps1, f1)
  expect_equal(res1, list(merror = "Missing parameters: y"))

  # captures extra parameters
  f2 = function() NULL
  ps2 = ps(x = p_uty())

  res2 = autotest_paramset(ps2, f2)
  expect_equal(res2, list(eerror = "Extra parameters: x"))

  # defaults
  f3 = function(x = 1, y = 1, z) NULL
  ps3 = ps(x = p_int(default = 2), y = p_uty(), z = p_uty(default = 3))

  res3 = autotest_paramset(ps3, f3)
  expect_equal(res3, list(derror = "Wrong defaults: x, y, z"))

  # Excluding stuff works

  # Can exclude function arguments
  f4 = function(x, y) NULL
  ps4 = ps(x = p_uty())

  res4 = autotest_paramset(ps4, f4, exclude = "y")
  expect_equal(res4, list())

  # Can exclude parameters
  f5 = function(x) NULL
  ps5 = ps(x = p_uty(), y = p_uty())

  res5 = autotest_paramset(ps5, f5, exclude = "y")
  expect_equal(res5, list())

  # Defaults of exlcuded parameters are not checked
  f6 = function(x) NULL
  ps6 = ps(x = p_uty(), y = p_uty(default = 10))

  res6 = autotest_paramset(ps6, f6, exclude = "y")
  expect_equal(res6, list())

  # Exclude defaults works
  f7 = function(x = 10) NULL
  ps7 = ps(x = p_uty())
  res7 = autotest_paramset(ps7, f7, exclude_defaults = "x")
  expect_equal(res7, list())

  f8 = function(x) NULL
  ps8 = ps(x = p_uty(default = 3))

  res8 = autotest_paramset(ps8, f8, exclude_defaults = "x")
  expect_equal(res8, list())

  # Multiple functions work as well

  f9 = list(function(x) NULL, function(y) NULL)
  ps9 = ps(x = p_uty(default = 3), z = p_uty())

  res9 = autotest_paramset(ps9, f9)
  expect_equal(res9, list(merror = "Missing parameters: y", eerror = "Extra parameters: z",
    derror = "Wrong defaults: x"))

  # Works for argument NULL
  f10 = function(x = NULL) NULL
  ps10 = ps(x = p_uty(default = NULL))

  res10 = autotest_paramset(ps10, f10)
  expect_equal(res10, list())

  f11 = f10
  ps11 = ps(x = p_uty())
  res11 = autotest_paramset(ps11, f11)

  # negative values work (evaluation problem, see comments in autotest_paramset)
  f12 = function(x = -1, y = c("a", "b")) NULL
  ps12 = ps(x = p_int(default = -1), y = p_uty(default = c("a", "b")))
  res12 = autotest_paramset(ps12, f12)
  expect_equal(res12, list())

  f13 = f12
  ps13 = ps(x = p_int(), y = p_uty())
  res13 = autotest_paramset(ps13, f13)
  expect_equal(res13, list(derror = "Wrong defaults: x, y"))

  # works if expression cannot be evaluated
  f14 = function(x = z, z = 1) NULL
  ps14 = ps(x = p_uty(), z = p_int(default = 1))
  res14 = autotest_paramset(ps14, f14)
  expect_equal(res14, list(derror = "Wrong defaults: x"))
})
