test_ingress = function(po_ingress, task) {
  expect_error(po_ingress$train(list(task))[[1]], regexp = "Task contains features of type")

  # selecting the feature types first is the same as setting select to TRUE
  tmp = po("select", selector = selector_type(c(po_ingress$feature_types)))$train(list(task))
  token_presel = po_ingress$train(tmp)[[1L]]
  po_ingress$param_set$values$select = TRUE
  token = po_ingress$train(list(task))[[1]]
  expect_equal(token, token_presel)

  expect_true(token$graph$ids() == po_ingress$id)
  expect_true(all(token$task$feature_types$type %in% po_ingress$feature_types))
  expect_equal(token$callbacks, list())
  expect_equal(token$.pointer, c(po_ingress$id, "output"))
  expect_equal(token$.pointer_shape, get_private(po_ingress)$.shape(task, po_ingress$param_set$values))

  ingress = token$ingress
  expect_set_equal(
    ingress[[1L]]$features,
    task$feature_types[get("type") %in% po_ingress$feature_types, "id", with = FALSE][[1L]]
  )

  ds = task_dataset(task, ingress, device = "cpu")
  batch = ds$.getbatch(1)
  x = batch$x[[1L]]
  expect_true(torch_equal(x, token$ingress[[1L]]$batchgetter(task$data(1, token$task$feature_names), "cpu")))
}

test_that("PipeOpTorchIngressNumeric", {
  po_ingress = po("torch_ingress_num")
  task = tsk("penguins")
  test_ingress(po_ingress, task)
})

test_that("PipeOpTorchIngressCategorical", {
  po_ingress = po("torch_ingress_cat")
  dat = data.table(y = runif(10), x_cat = factor(letters[1:10]), x_lgl = TRUE, x_ord = ordered(letters[1:10]),
    x_num = runif(10), x_int = 1:10
  )
  task = as_task_regr(dat, target = "y")
  test_ingress(po_ingress, task)
})

test_that("PipeOpTorchIngressImage", {
  po_ingress = po("torch_ingress_img", channels = 3, width = 64, height = 64)
  task = toytask()$cbind(data.frame(x1 = 1:200))
  test_ingress(po_ingress, task)
})
