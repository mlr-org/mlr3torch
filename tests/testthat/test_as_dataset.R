test_that("as_dataset works", {
  task = tsk("iris")
  ds = as_dataset(task)
  expect_r6(task, "Dataset")
})
