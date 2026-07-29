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
  # as.data.table() constructs every task in the dictionary, and 'pima' currently fails to
  # construct because mlbench no longer ships PimaIndiansDiabetes2. Drop it for this test and
  # put it back afterwards.
  # TODO: remove once this is fixed upstream in mlr3
  if ("pima" %in% mlr_tasks$keys()) {
    pima = mlr_tasks$items[["pima"]]
    mlr_tasks$remove("pima")
    withr::defer(mlr_tasks$add("pima", pima$value, .prototype_args = pima$prototype_args))
  }
  expect_data_table(as.data.table(mlr_tasks))
  # nothing is in the cache directory -> imagenet was not downloaded
  expect_true(!dir.exists(dir))
})
