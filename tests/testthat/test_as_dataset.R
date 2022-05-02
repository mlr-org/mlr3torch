test_that("as_dataset works for tabular data", {
  task = tsk("iris")
  ds = as_dataset(task, row_ids = 1:10)
  expect_true(length(ds) == 10L)
  expect_r6(ds, "dataset")
})

test_that("as_dataset works for image data", {
  task = tsk("tiny_imagenet")
  ds = as_dataset(task)
  expect_true(inherits(ds, "dataset"))
  batch = ds$.getbatch(1:16)
  expect_true(all.equal(names(batch), c("y", "x")))
  y = batch$y
  x = batch$x
  expect_true(inherits(y, "torch_tensor"))
  expect_true(inherits(x, "torch_tensor"))
  expect_true(x$shape[1] == 16)
  expect_true(y$shape[1] == 16)
})
