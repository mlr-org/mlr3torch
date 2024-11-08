skip_on_cran()

test_that("melanoma task works", {
  withr::local_options(mlr3torch.cache = TRUE)
  task = tsk("melanoma")
  # this makes the test faster
  task$row_roles$use = 1:10
  expect_equal(task$id, "melanoma")
  expect_equal(task$label, "Melanoma classification")
  expect_equal(task$feature_names, "image")
  expect_equal(task$target_names, "target")
  expect_equal(task$man, "mlr3torch::mlr_tasks_melanoma")
  expect_equal(task$properties, "twoclass")

  x = materialize(task$data(task$row_ids[1:2], cols = "image")[[1L]], rbind = TRUE)
  expect_equal(x$shape, c(2, 3, 128, 128))
  expect_equal(x$dtype, torch_float32())
})