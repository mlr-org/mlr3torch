skip_on_cran()

test_that("mnist task works", {
  withr::local_options(mlr3torch.cache = TRUE)
  task = tsk("mnist")
  expect_equal(task$row_roles$use, 1:60000)
  expect_equal(task$row_roles$test, 60001:70000)
  # this makes the test faster
  task$row_roles$use = 1:10
  expect_equal(task$id, "mnist")
  expect_equal(task$label, "MNIST Digit Classification")
  expect_equal(task$feature_names, "image")
  expect_equal(task$target_names, "label")
  expect_equal(task$man, "mlr3torch::mlr_tasks_mnist")
  expect_equal(task$properties, "multiclass")
})
