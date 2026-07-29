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
  expect_equal(get_batch_size(list(batch_size = 16), "train"), 16)
  expect_equal(get_batch_size(list(batch_size = 16), "predict"), 16)
  expect_equal(get_batch_size(list(batch_size = 16, batch_size_predict = 32), "train"), 16)
  expect_equal(get_batch_size(list(batch_size = 16, batch_size_predict = 32), "predict"), 32)
  expect_null(get_batch_size(list(batch_size_predict = 32), "train"))
  expect_null(get_batch_size(list(), "predict"))
  expect_null(get_batch_size(list(batch_size = NULL), "train"))

  # arguments are asserted
  expect_error(get_batch_size(16, "train"), "list")
  expect_error(get_batch_size(list(batch_size = 16), "valid"), "element of set")
  expect_error(get_batch_size(list(batch_size = 0), "train"), ">= 1")
  expect_error(get_batch_size(list(batch_size = c(16, 32)), "train"), "length 1")
  expect_error(get_batch_size(list(batch_size = "16"), "train"), "integerish")
})

test_that("make_check_class works", {
  check = make_check_class("torch_sampler")
  sampler = torch::sampler("S",
    initialize = function(data_source) NULL,
    .iter = function() function() coro::exhausted(),
    .length = function() 0L
  )
  expect_true(check(sampler))
  expect_string(check(1))
  expect_string(check(sampler(1)))
  expect_error(make_check_class(1), "character")
})
