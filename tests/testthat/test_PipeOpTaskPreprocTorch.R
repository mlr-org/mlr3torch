test_that("PipeOpTaskPreprocTorch: basic checks", {
  po_test = po("preproc_torch", identity, packages = "R6")
  expect_pipeop(po_test)
  expect_equal(po_test$feature_types, "lazy_tensor")
  expect_true(po_test$innum == 1L)
  expect_true(po_test$outnum == 1L)
  expect_class(po_test, "PipeOpTaskPreproc")
  expect_true("R6" %in% po_test$packages)
  expect_set_equal(po_test$param_set$ids(), c("augment", "affect_columns"))
  expect_error(po("preproc_torch"), "is missing")

  PipeOpTaskPreprocTorchTest = R6Class("PipeOpTaskPreprocTorchTest",
    inherit = PipeOpTaskPreprocTorch,
    public = list(
      initialize = function(id = "test1", param_vals = list()) {
        super$initialize(
          id = id,
          param_vals = param_vals,
          fn = function(x) -x
        )
      }
    ),
    private = list(
      # check that shapes_out calls `$.shapes_out()` on each lazy tensor column
      .shapes_out = function(shapes_in, param_vals, task) list(shapes_in[[1L]])
    )
  )

  po_test1 = PipeOpTaskPreprocTorchTest$new(id = "test1")
  expect_true("augment" %in% po_test1$param_set$ids())

  shapes_in = list(c(NA, 1), c(NA, 1))
  expect_identical(shapes_in, po_test1$shapes_out(shapes_in, stage = "train"))
  expect_identical(shapes_in, po_test1$shapes_out(shapes_in, stage = "predict"))
  shapes_in1 = list(a = c(NA, 1), b = c(NA, 1))

  expect_true(is.null(names(po_test1$shapes_out(shapes_in1, "train"))))

  task = as_task_regr(data.table(
    y = 1:10,
    x1 = as_lazy_tensor(as.double(1:10)),
    x2 = as_lazy_tensor(as.double(1:10))
  ), target = "y")

  po_test1$param_set$set_values(
    affect_columns = selector_name("x1")
  )

  taskout1 = po_test1$train(list(task))[[1L]]
  expect_torch_equal(
    -materialize(taskout1$data(cols = "x1")[[1L]]),
    materialize(taskout1$data(cols = "x2")[[1L]])
  )

  # "train" and "predict" tags are respected
  PipeOpTaskPreprocTorchTest2 = R6Class("PipeOpTaskPreprocTorchTest2",
    inherit = PipeOpTaskPreprocTorch,
    public = list(
      initialize = function(id = "test2", param_vals = list()) {
        param_set = ps(
          a = p_int(tags = "train")
        )
        super$initialize(
          param_set = param_set,
          id = id,
          param_vals = param_vals,
          fn = function(x, a = 2) x * a
        )

      }
    )
  )

  po_test2 = PipeOpTaskPreprocTorchTest2$new(param_vals = list(affect_columns = selector_name("x1"), a = 0))

  expect_true(
    torch_sum(materialize(po_test2$train(list(task))[[1L]]$data(cols = "x1")[[1L]]))$item() == 0
  )

  expect_true(
    torch_sum(materialize(po_test2$predict(list(task))[[1L]]$data(cols = "x1")[[1L]]))$item() == 2 * sum(1:10)
  )

  # augment parameter works as intended (above augment was FALSE)

  po_test2$param_set$set_values(
    augment = TRUE
  )
  expect_true(
    torch_sum(materialize(po_test2$predict(list(task))[[1L]]$data(cols = "x1")[[1L]]))$item() == sum(1:10)
  )
})

test_that("PipeOpTaskPreprocTorch modifies the underlying lazy tensor columns correctly", {
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

test_that("pipeop_preproc_torch works", {
  expect_error(
    pipeop_preproc_torch("abc", function(x) NULL, shapes_out = function(shapes) NULL),
    "Must have formal"
  )

  po_test = pipeop_preproc_torch("abc", function(x, a) torch_cat(list(x, torch_pow(x, a)), dim = 2),
    shapes_out = function(shapes_in, param_vals, task) {
      s = shapes_in[[1L]]
      s[2] = 2L
      s
      list(s)
    }
  )

  expect_class(po_test, "PipeOpPreprocTorchAbc")
  # parameter a was added
  expect_set_equal(c("a", "affect_columns", "augment"), po_test$param_set$ids())

  task = as_task_regr(data.table(
    y = 2,
    x = as_lazy_tensor(3)
  ), target = "y")

  po_test$param_set$set_values(
    a = 0
  )

  taskout = po_test$train(list(task))[[1L]]

  x = materialize(taskout$data(cols = "x")[[1L]])
  expect_torch_equal(x[1, 1]$item(), 3)
  expect_torch_equal(x[1, 2]$item(), 1)

  po_test1 = pipeop_preproc_torch("test1", torchvision::transform_resize, shapes_out = TRUE,
    param_vals = list(size = c(10, 10))
  )

  po_test$shapes_out(list(c(NA, 20, 20)), "train")
})

test_that("predict shapes are added during training", {
  po_test = pipeop_preproc_torch("test", function(x) torch_cat(list(x, x * 2), dim = 2))
  task = as_task_regr(data.table(
    y = 1,
    x = as_lazy_tensor(1)
  ), target = "y")

  taskout = po_test$train(list(task))[[1L]]

})
