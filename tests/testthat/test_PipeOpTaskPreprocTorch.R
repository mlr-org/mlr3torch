test_that("PipeOpTaskPreprocTorch: basic checks", {
  po_test = po("preproc_torch", identity, packages = "R6")
  expect_pipeop(po_test)
  expect_equal(po_test$feature_types, "lazy_tensor")
  expect_true(po_test$innum == 1L)
  expect_true(po_test$outnum == 1L)
  expect_class(po_test, "PipeOpTaskPreproc")
  expect_true("R6" %in% po_test$packages)
  expect_set_equal(po_test$param_set$ids(), c("stages", "affect_columns"))
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
  expect_true("stages" %in% po_test1$param_set$ids())

  shapes_in = list(c(NA, 1), c(NA, 1))
  expect_identical(shapes_in, po_test1$shapes_out(shapes_in, stage = "train"))
  expect_error(po_test1$shapes_out(shapes_in, stage = "predict"), "can only be calculated")

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
    -materialize(taskout1$data(cols = "x1")[[1L]], rbind = TRUE),
    materialize(taskout1$data(cols = "x2")[[1L]], rbind = TRUE)
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
    torch_sum(materialize(po_test2$train(list(task))[[1L]]$data(cols = "x1")[[1L]], rbind = TRUE))$item() == 0
  )

  expect_true(
    torch_sum(materialize(po_test2$predict(list(task))[[1L]]$data(cols = "x1")[[1L]], rbind = TRUE))$item() == 0
  )

  # stages parameter works as intended (above stages was set to "both")

  po_test2$param_set$set_values(
    stages = "train"
  )
  # stages is c("train", "predict"), but a is tagged "train" --> does not change the results
  expect_true(
    torch_sum(materialize(po_test2$predict(list(task))[[1L]]$data(cols = "x1")[[1L]], rbind = TRUE))$item() == 0
  )
  # only if we re-train the transformatoin is applied during predict
  po_test2$train(list(task))
  expect_true(
    torch_sum(materialize(po_test2$predict(list(task))[[1L]]$data(cols = "x1")[[1L]], rbind = TRUE))$item() == sum(1:10)
  )

  # when stages is set to "predict", it onlu applied during prediction
  po_test$param_set$set_values(stages = "predict")
  expect_torch_equal(
    materialize(task$data(cols = "x1")[[1L]], rbind = TRUE),
    materialize(po_test$train(list(task))[[1L]]$data(cols = "x1")[[1L]], rbind = TRUE)

  )
  # TODO:
  # need to finish the augment -> stages transition, i.e. add tests and "both", then finish the preprocess impmenetation. add the tests and test the preprocess autotest
  # also test that the rowwise parameter works

  materialize(task$data(cols = "x1")[[1L]])

})

test_that("PipeOptaskPreprocTorch: shapes_out() works", {
  task = nano_dogs_vs_cats()
  po_resize = po("trafo_resize", size = c(10, 10))
  expect_identical(po_resize$shapes_out(list(x = NULL), stage = "train"), list(NULL))
  expect_identical(po_resize$shapes_out(list(x = NULL, y = c(NA, 3, 5, 5)), stage = "train"), list(NULL, c(NA, 3, 10, 10)))
  expect_error(po_resize$shapes_out(list(x = c(NA, 1, 3)), stage = "predict"), "can only be calculated")

  # predict when stages is "train"
  po_resize$param_set$set_values(stages = "train")
  po_resize$train(list(task))
  expect_identical(po_resize$shapes_out(list(x = NULL), stage = "predict"), list(NULL))
  expect_identical(po_resize$shapes_out(list(x = NULL, y = c(NA, 3, 5, 5)), stage = "predict"), list(NULL, c(NA, 3, 5, 5)))

  # predict when stages is c("train", "predict"")
  po_resize$param_set$set_values(stages = "both")
  po_resize$train(list(task))
  expect_identical(po_resize$shapes_out(list(x = NULL), stage = "predict"), list(NULL))
  expect_identical(po_resize$shapes_out(list(x = NULL, y = c(NA, 3, 5, 5)), stage = "predict"), list(NULL, c(NA, 3, 10, 10)))
})

test_that("PipeOpTaskPreprocTorch modifies the underlying lazy tensor columns correctly", {
  d = data.table(
    y = 1:10,
    x = as_lazy_tensor(rnorm(10))
  )

  taskin = as_task_regr(d, target = "y")

  po_test = po("preproc_torch", fn = crate(function(x, a) x + a), param_set = ps(a = p_int(tags = c("train", ""))),
    a = -10, rowwise = FALSE)

  taskout_train = po_test$train(list(taskin))[[1L]]
  taskout_pred = po_test$predict(list(taskin))[[1L]]

  po_test$param_set$set_values(stages = "train")
  po_test$train(list(taskin))[[1L]]
  taskout_pred_aug = po_test$predict(list(taskin))[[1L]]

  expect_true(length(taskin$data(cols = "x")[[1]]$graph$pipeops) == 1L)

  expect_true(length(taskout_pred$data(cols = "x")[[1]]$graph$pipeops) == 2L)
  expect_true(length(taskout_pred_aug$data(cols = "x")[[1]]$graph$pipeops) == 2L)

  expect_true(identical(taskout_pred_aug$data(cols = "x")[[1]]$graph$pipeops[[2]]$module, identity))
  expect_false(identical(taskout_pred$data(cols = "x")[[1]]$graph$pipeops[[2]]$module, identity))

  taskout_pred$data(cols = "x")[[1]]$graph

  expect_equal(
    as_array(materialize(taskin$data(cols = "x")[[1L]], rbind = TRUE)),
    as_array(materialize(taskout_train$data(cols = "x")[[1L]], rbind = TRUE) + 10),
    tolerance = 1e-5
  )
})

test_that("pipeop_preproc_torch", {
  expect_error(
    pipeop_preproc_torch("trafo_abc", function(x) NULL, shapes_out = function(shapes) NULL),
    "Must have formal arguments"
  )

  po_test = pipeop_preproc_torch("trafo_abc", function(x, a) torch_cat(list(x, torch_pow(x, a)), dim = 2),
    shapes_out = function(shapes_in, param_vals, task) {
      s = shapes_in[[1L]]
      s[2] = 2L
      s
      list(s)
    }
  )

  expect_true("required" %in% po_test$param_set$tags$a)
  expect_class(po_test, "PipeOpPreprocTorchTrafoAbc")
  # parameter a was added
  expect_set_equal(c("a", "affect_columns", "stages"), po_test$param_set$ids())

  task = as_task_regr(data.table(
    y = 2,
    x = as_lazy_tensor(3)
  ), target = "y")

  po_test$param_set$set_values(
    a = 0
  )

  taskout = po_test$train(list(task))[[1L]]

  x = materialize(taskout$data(cols = "x")[[1L]], rbind = TRUE)
  expect_torch_equal(x[1, 1]$item(), 3)
  expect_torch_equal(x[1, 2]$item(), 1)

  po_test1 = pipeop_preproc_torch("test1", torchvision::transform_resize, shapes_out = "infer",
    param_vals = list(size = c(10, 10))
  )

  size = po_test1$shapes_out(list(c(NA, 20, 20)), "train")
  expect_equal(size, list(c(NA, 10, 10)))
})

test_that("predict shapes are added during training", {
  po_test = pipeop_preproc_torch("test", function(x) torch_cat(list(x, x * 2), dim = 2), shapes_out = "infer")
  task = as_task_regr(data.table(
    y = 1,
    x = as_lazy_tensor(1)
  ), target = "y")

  taskout = po_test$train(list(task))[[1L]]

})

