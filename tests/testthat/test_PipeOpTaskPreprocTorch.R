test_that("Correct errors", {


})

test_that("PipeOpTaskPreprocTorch works with one lazy tensor column", {
  d = data.table(
    y = 1:10,
    x = as_lazy_tensor(rnorm(10))
  )

  taskin = as_task_regr(d, target = "y")

  po_test = po("preproc_torch", fn = crate(function(x, a) x + a), param_set = ps(a = p_int(tags = c("train", ""))), a = -10)

  taskout_train = po_test$train(list(taskin))[[1L]]
  taskout_pred = po_test$predict(list(taskin))[[1L]]

  po_test$param_set$set_values(augment = TRUE)
  taskout_pred_aug = po_test$predict(list(taskin))[[1L]]

  expect_true(length(taskin$data(cols = "x")[[1]]$graph$pipeops) == 1L)

  expect_true(length(taskout_pred$data(cols = "x")[[1]]$graph$pipeops) == 2L)
  expect_true(length(taskout_pred_aug$data(cols = "x")[[1]]$graph$pipeops) == 2L)

  expect_true(identical(taskout_pred_aug$data(cols = "x")[[1]]$graph$pipeops[[2]]$module, identity))
  expect_false(identical(taskout_pred$data(cols = "x")[[1]]$graph$pipeops[[2]]$module, identity))

  taskout_pred$data(cols = "x")[[1]]$graph

  expect_equal(
    as_array(materialize(taskin$data(cols = "x")[[1L]])),
    as_array(materialize(taskout_train$data(cols = "x")[[1L]]) + 10),
    tolerance = 1e-5
  )
})

test_that("PipeOpTaskPreprocTorch works with multiple lazy tensor columns")
