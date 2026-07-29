test_that("paramset works", {
  test_ps = function(param_set) {
    expect_r6(param_set, "ParamSet")
    expect_true(all(map_lgl(param_set$tags, function(tags) "train" %in% tags || "predict" %in% tags)))
  }
  param_set_regr = paramset_torchlearner("regr")

  test_ps(param_set_regr)

  expect_error(param_set_regr$set_values(measures_train = msr("regr.mse")), regexp = NA)
  expect_error(param_set_regr$set_values(measures_valid = msr("regr.mse")), regexp = NA)

  expect_error(param_set_regr$set_values(measures_train = msr("classif.acc")), "regr")
  expect_error(param_set_regr$set_values(measures_valid = msr("classif.acc")), "regr")

  param_set_classif = paramset_torchlearner("classif")
  test_ps(param_set_classif)
  expect_error(param_set_classif$set_values(measures_train = msr("classif.acc")), regexp = NA)
  expect_error(param_set_classif$set_values(measures_valid = msr("classif.acc")), regexp = NA)
  expect_error(param_set_classif$set_values(measures_train = msr("regr.mse")), regexp = "classif")
  expect_error(param_set_classif$set_values(measures_valid = msr("regr.mse")), regexp = "classif")
  expect_error(param_set_classif$set_values(measures_train = msr("selected_features")), regexp = "must not require")


  expect_error({param_set_regr$values$device = "opengl"}, regexp = NA) # nolint
})

test_that("make_check_measures works", {
  expect_true(check_measures_regr(msr("regr.mse")))
  expect_true(check_measures_regr(list(msr("regr.mse"))))
  expect_true(check_measures_regr(msrs(c("regr.mse", "regr.mae"))))
  expect_grepl_regr = function(x, pattern) expect_true(grepl(pattern, check_measures_regr(x)))
  expect_grepl_regr(msrs(c("regr.mse", "regr.mse")), "IDs of measures")
  expect_grepl_regr(msrs(c("regr.mse", "classif.acc")), "regr")
  # cannot have property "requires_model"
  expect_grepl_regr(msrs(c("oob_error")), "require a learner or model")
  # has property "requires_learner"
  expect_grepl_regr(msrs(c("time_predict")), "require a learner or model")

  expect_grepl_classif = function(x, pattern) expect_true(grepl(pattern, check_measures_classif(x)))
  expect_true(check_measures_classif(msr("classif.acc")))
  expect_true(check_measures_classif(list(msr("classif.acc"))))
  expect_true(check_measures_classif(msrs(c("classif.acc", "classif.ce"))))
  expect_grepl_classif(msrs(c("regr.mse", "classif.acc")), "classif")
  # cannot have property "requires_model"
  expect_grepl_classif(msrs(c("oob_error")), "require a learner or model")
  # has property "requires_learner"
  expect_grepl_classif(msrs(c("time_predict")), "require a learner or model")
})

test_that("get_batch_size works", {
  expect_equal(get_batch_size(16, "train"), 16)
  expect_equal(get_batch_size(16, "predict"), 16)
  expect_equal(get_batch_size(c(train = 16, predict = 32), "train"), 16)
  expect_equal(get_batch_size(c(train = 16, predict = 32), "predict"), 32)
  expect_null(get_batch_size(c(predict = 32), "train"))
  expect_null(get_batch_size(c(train = 16), "predict"))
  expect_null(get_batch_size(NULL, "train"))
  # the phase name is dropped
  expect_null(names(get_batch_size(c(train = 16), "train")))
})

test_that("check_batch_size works", {
  expect_true(check_batch_size(16))
  expect_true(check_batch_size(1L))
  expect_true(check_batch_size(c(train = 16)))
  expect_true(check_batch_size(c(predict = 32)))
  expect_true(check_batch_size(c(train = 16, predict = 32)))

  expect_grepl = function(x) expect_true(grepl("positive integer", check_batch_size(x)))
  expect_grepl(0)
  expect_grepl(-1)
  expect_grepl(1.5)
  expect_grepl("16")
  expect_grepl(NA_integer_)
  expect_grepl(integer(0))
  expect_grepl(c(16, 32))
  expect_grepl(c(train = 16, train = 32))
  expect_grepl(c(train = 16, foo = 32))
  expect_grepl(c(train = 16, predict = 32, foo = 8))
})
