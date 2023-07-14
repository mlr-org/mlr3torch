test_that("ResamplingRowRoles works", {
  resampling = rsmp("row_roles")
  resampling

  expect_class(resampling, c("ResamplingRowRoles", "Resampling"))
  expect_equal(resampling$iters, 1)
  expect_true(resampling$duplicated_ids)

  expect_man_exists("mlr3torch::mlr_resamplings_row_roles")

  expect_error(resampling$test_set(1), "not been instantiated")
  expect_error(resampling$train_set(1), "not been instantiated")

  task = tsk("german_credit")
  splits = partition(task)
  task$set_row_roles(splits$train, "use")
  task$set_row_roles(splits$test, "test")
  task$set_col_roles("telephone", "group")

  expect_error(resampling$instantiate(task), "cannot be ensured")
  task$set_col_roles("telephone", "stratum")
  expect_error(resampling$instantiate(task), "cannot be ensured")

  task$col_roles$stratum = character(0)
  resampling$instantiate(task)
  expect_equal(resampling$task_hash, task$hash)
  expect_equal(resampling$task_nrow, task$nrow)

  expect_equal(resampling$train_set(1), task$row_roles$use)
  expect_equal(resampling$test_set(1), task$row_roles$test)

  rr = resample(task, lrn("classif.featureless"), resampling)

  pred_test = as.data.table(rr$prediction("test"))
  expect_permutation(pred_test$row_ids, task$row_roles$test)

  pred_train = as.data.table(rr$prediction("train"))
  expect_permutation(pred_train$row_ids, task$row_toles$use)
})
