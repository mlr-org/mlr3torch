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
