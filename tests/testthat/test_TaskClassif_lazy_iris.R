test_that("tiny_imagenet task works", {
  task = tsk("lazy_iris")
  expect_task(task)
  expect_equal(task$id, "lazy_iris")
  expect_equal(task$label, "Iris Flowers")
  expect_equal(task$target_names, "Species")
  expect_equal(task$feature_names, "x")
  expect_equal(task$man, "mlr3torch::mlr_tasks_lazy_iris")
  expect_equal(task$nrow, 150)
  expect_equal(task$ncol, 2)
})

