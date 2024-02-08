test_that("mlr_pipeops can be converted to a table", {
  tbl = as.data.table(mlr_pipeops)
  expect_data_table(tbl)
})

test_that("mlr_learners can be converted to a table", {
  tbl = as.data.table(mlr_learners)
  expect_data_table(tbl)
})

test_that("mlr3torch_callbacks can be converted to a table", {
  tbl = as.data.table(mlr3torch_callbacks)
  expect_data_table(tbl)
})

test_that("mlr3torch_optimizers can be converted to a table", {
  tbl = as.data.table(mlr3torch_optimizers)
  expect_data_table(tbl)
})

test_that("mlr3torch_losses can be converted to a table", {
  tbl = as.data.table(mlr3torch_losses)
  expect_data_table(tbl)
})

test_that("mlr_tasks can be converted to a table without downloading imagenet", {
  dir = tempfile()
  withr::local_options(mlr3torch.cache = dir)
  expect_data_table(as.data.table(mlr_tasks))
  # nothing is in the cache directory -> imagenet was not downloaded
  expect_true(!dir.exists(dir))
})
