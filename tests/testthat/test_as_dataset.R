test_that("as_dataset works", {
  task = tsk("iris")
  ds = as_dataset(task, row_ids = 1:10)
  expect_true(length(ds) == 10L)
  expect_r6(ds, "dataset")
})
