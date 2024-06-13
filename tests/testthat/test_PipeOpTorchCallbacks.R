test_that("Basic checks", {
  tcb = t_clbk("checkpoint")
  tcb$packages = union(tcb$packages, "stats")
  obj = po("torch_callbacks", list(tcb, t_clbk("progress")), checkpoint.path = "abc")

  task = tsk("iris")
  md = po("torch_ingress_num")$train(list(task))
  mdout = obj$train(md)[[1L]]

  expect_subset("stats", obj$packages)
  expect_true(obj$is_trained)
  expect_identical(obj$state, list())
  expect_pipeop(obj)
  expect_list(mdout$callbacks, types = "TorchCallback")
  expect_equal(ids(mdout$callbacks), c("checkpoint", "progress"))
  expect_equal(mdout$callbacks$checkpoint$param_set$values$path, "abc")

  expect_error(po("torch_callbacks", 1:2))
})

test_that("Repeated application works", {
  obj1 = po("torch_callbacks_1", callbacks = "checkpoint")
  obj2 = po("torch_callbacks_2", callbacks = "progress")
  md = po("torch_ingress_num")$train(list(tsk("iris")))[[1L]]

  graph = obj1 %>>% obj2
  mdout = graph$train(md)[[1L]]

  expect_true(length(mdout$callbacks) == 2L)
  expect_set_equal(ids(mdout$callbacks), c("checkpoint", "progress"))

  obj3 = po("torch_callbacks_2", callbacks = "checkpoint")

  graph2 = obj1 %>>% obj3
  expect_error(graph2$train(md),
    fixed = TRUE,
    regexp = "Callbacks with IDs 'checkpoint' are already present.",
  )

  # doing nothing twice works
  obj4 = po("torch_callbacks_1")
  obj5 = po("torch_callbacks_2")

  graph3 = obj4 %>>% obj5

  mdout2 = graph3$train(md)[[1L]]
  expect_identical(mdout2$callbacks, list())
})


test_that("The callbacks are cloned during construction", {
  tclbks = t_clbks(c("progress", "checkpoint"))

  obj = po("torch_callbacks", tclbks)
  expect_true(address(tclbks[[1L]]) != address(get_private(obj)$.callbacks[[1L]]))
  expect_true(address(tclbks[[2L]]) != address(get_private(obj)$.callbacks[[2L]]))
})

test_that("Cloning works", {
  obj1 = po("torch_callbacks", callbacks = c("progress", "history"))
  obj2 = obj1$clone(deep = TRUE)
  expect_deep_clone(obj1, obj2)
})
