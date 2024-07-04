skip_on_cran()

test_that("mnist task works", {
  withr::local_options(mlr3torch.cache = TRUE)
  task = tsk("mnist")
  # this makes the test faster
  task$row_roles$use = 1:10
  expect_equal(task$id, "mnist")
  expect_equal(task$label, "MNIST Digit Classification")
  expect_equal(task$feature_names, "image")
  expect_equal(task$target_names, "label")
  expect_equal(task$man, "mlr3torch::mlr_tasks_mnist")
  expect_equal(task$properties, "multiclass")

  x = materialize(task$data(task$row_ids[1:2], cols = "image")[[1L]], rbind = TRUE)
  expect_equal(x$shape, c(2, 1, 28, 28))
  expect_equal(x$dtype, torch_float32())
})
