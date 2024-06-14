test_that("auto_paramtest works", {
  # captures missing parameters
  f1 = function(x, y) NULL
  ps1 = ps(x = p_uty())
  res1 = expect_paramset(ps1, f1)
  expect_equal(res1$info, list(merror = "Missing parameters: y"))
  expect_false(res1$ok)

  # captures extra parameters
  f2 = function() NULL
  ps2 = ps(x = p_uty())

  res2 = expect_paramset(ps2, f2)
  expect_equal(res2$info, list(eerror = "Extra parameters: x"))

  # defaults
  f3 = function(x = 1, y = 1, z) NULL
  ps3 = ps(x = p_int(default = 2), y = p_uty(), z = p_uty(default = 3))

  res3 = expect_paramset(ps3, f3)
  expect_equal(res3$info, list(derror = "Wrong defaults: x, y, z"))

  # Excluding stuff works

  # Can exclude function arguments
  f4 = function(x, y) NULL
  ps4 = ps(x = p_uty())

  res4 = expect_paramset(ps4, f4, exclude = "y")
  expect_equal(res4$info, list())
  expect_true(res4$ok)

  # Can exclude parameters
  f5 = function(x) NULL
  ps5 = ps(x = p_uty(), y = p_uty())

  res5 = expect_paramset(ps5, f5, exclude = "y")
  expect_equal(res5$info, list())

  # Defaults of exlcuded parameters are not checked
  f6 = function(x) NULL
  ps6 = ps(x = p_uty(), y = p_uty(default = 10))

  res6 = expect_paramset(ps6, f6, exclude = "y")
  expect_equal(res6$info, list())

  # Exclude defaults works
  f7 = function(x = 10) NULL
  ps7 = ps(x = p_uty())
  res7 = expect_paramset(ps7, f7, exclude_defaults = "x")
  expect_equal(res7$info, list())

  f8 = function(x) NULL
  ps8 = ps(x = p_uty(default = 3))

  res8 = expect_paramset(ps8, f8, exclude_defaults = "x")
  expect_equal(res8$info, list())

  # Multiple functions work as well

  f9 = list(function(x) NULL, function(y) NULL)
  ps9 = ps(x = p_uty(default = 3), z = p_uty())

  res9 = expect_paramset(ps9, f9)
  expect_equal(res9$info, list(merror = "Missing parameters: y", eerror = "Extra parameters: z",
    derror = "Wrong defaults: x"))

  # Works for argument NULL
  f10 = function(x = NULL) NULL
  ps10 = ps(x = p_uty(default = NULL))

  res10 = expect_paramset(ps10, f10)
  expect_equal(res10$info, list())

  f11 = f10
  ps11 = ps(x = p_uty())
  res11 = expect_paramset(ps11, f11)
  expect_equal(res11$info, list(derror = "Wrong defaults: x"))

  # negative values work (evaluation problem, see comments in expect_paramset)
  f12 = function(x = -1, y = c("a", "b")) NULL
  ps12 = ps(x = p_int(default = -1), y = p_uty(default = c("a", "b")))
  res12 = expect_paramset(ps12, f12)
  expect_equal(res12$info, list())

  f13 = f12
  ps13 = ps(x = p_int(), y = p_uty())
  res13 = expect_paramset(ps13, f13)
  expect_equal(res13$info, list(derror = "Wrong defaults: x, y"))

  # works if expression cannot be evaluated
  f14 = function(x = z, z = 1) NULL
  ps14 = ps(x = p_uty(), z = p_int(default = 1))
  res14 = expect_paramset(ps14, f14)
  expect_equal(res14$info, list(derror = "Wrong defaults: x"))

  # Should not trigger default message if the parameter is missing or extra
  f15 = function(x = 1) NULL
  ps15 = ps()
  res15 = expect_paramset(ps15, f15)
  expect_equal(res15$info, list(merror = "Missing parameters: x"))

  f16 = function() NULL
  ps16 = ps(x = p_int(default = 1))
  res16 = expect_paramset(ps16, f16)
  expect_equal(res16$info, list(eerror = "Extra parameters: x"))
})

test_that("expect_torch_callback works", {
  # captures misspelled stages
  CallbackSetA = R6Class("CallbackSetA",
    inherit = CallbackSet,
    public = list(
      on_edn = function() NULL
    )
  )
  cba = as_torch_callback(CallbackSetA)
  expect_error(
    expect_torch_callback(cba, check_man = FALSE),
    regexp = "but has additional elements"
  )
  tmp = class(cba)
  class(cba) = "blabla"
  expect_error(expect_torch_callback(cba, check_man = FALSE), "TorchCallback")
  class(cba) = tmp

  tmp = cba$id
  cba$id = 1
  expect_error(expect_torch_callback(cba, check_man = FALSE), "Must be of type")
  cba$id = tmp

  tmp = cba$label
  cba$label = 1.2
  expect_error(expect_torch_callback(cba, check_man = FALSE), "Must be of type")
  cba$label = tmp

  tmp = cba$generator
  cba$generator = 1
  expect_error(expect_torch_callback(cba, check_man = FALSE), "Must inherit from class")
  cba$generator = tmp

  CallbackSetB = R6Class("CallbackSetB",
    inherit = CallbackSet,
    public = list(
      initialize = function(a = -1) {
        NULL
      },
      on_begin = function() NULL
    )
  )
  cbb = as_torch_callback(CallbackSetB)
  expect_error(expect_torch_callback(cbb, check_man = FALSE), regexp = "Wrong defaults")

  CallbackSetC = R6Class("CallbackSetC",
    inherit = CallbackSet,
    public = list(
      on_begin = function(ctx) NULL
    )
  )

  cbc = as_torch_callback(CallbackSetC)

  CallbackSetD = R6Class("CallbackSetD",
    inherit = CallbackSet,
    lock_objects = FALSE,
    public = list(
      on_begin = function() NULL
    ),
    private = list(
      deep_clone = function(name, value) 1
    )
  )
  cbd = as_torch_callback(CallbackSetD)
  expect_error(expect_torch_callback(cbd, check_man = FALSE), regexp = "all\\.equal")
})
