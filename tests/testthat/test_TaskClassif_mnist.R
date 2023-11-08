test_that("tiny_imagenet task works", {
  task = tsk("lazy_iris")
  expect_task(task)
  expect_equal(task$id == "lazy_iris")
  expect_equal(task$man == "mlr3torch::mlr_tasks_lazy_iris")
})
