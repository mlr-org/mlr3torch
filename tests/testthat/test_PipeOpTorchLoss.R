test_that("Basic checks", {
  tloss = t_loss("mse")
  tloss$packages = union(tloss$packages, "stats")
  obj = po("torch_loss", loss = tloss, reduction = "sum")
  task = tsk("iris")

  md = po("torch_ingress_num")$train(list(task))
  mdout = obj$train(md)[[1L]]

  expect_equal(address(obj$param_set), address(get_private(obj)$.loss$param_set))

  expect_set_equal(tloss$param_set$ids(), obj$param_set$ids())
  expect_subset("stats", obj$packages)
  expect_true(obj$is_trained)
  expect_identical(obj$state, list())
  expect_pipeop(obj)
  expect_class(mdout$loss, "TorchLoss")
  expect_class(mdout$loss$generator, "nn_mse_loss")
  expect_equal(mdout$loss$param_set$values$reduction, "sum")

  expect_error(po("torch_loss", list()))
})

test_that("Correct error message if loss is already configured", {
  task = tsk("iris")
  md = po("torch_ingress_num")$train(list(task))

  obj = po("torch_loss", "cross_entropy")
  mdout = obj$train(md)

  expect_error(obj$train(mdout),
    regexp = "The loss of the model descriptor is already configured",
    fixed = TRUE
  )
})

test_that("The loss is cloned during construction", {
  tloss = t_loss("cross_entropy")

  obj = po("torch_loss", tloss)
  expect_true(address(tloss) != address(get_private(obj)$.loss))
})

test_that("Cloning works", {
  obj1 = po("torch_loss", "mse")
  obj2 = obj1$clone(deep = TRUE)
  expect_deep_clone(obj1, obj2)
  # parameter set references are preserved
  expect_equal(address(obj2$param_set), address(get_private(obj2)$.loss$param_set))

})

test_that("phash works", {
  expect_equal(
    po("torch_callbacks", list(t_clbk("history"), t_clbk("checkpoint", freq = 1)))$phash,
    po("torch_callbacks", list(t_clbk("history"), t_clbk("checkpoint", freq = 2)))$phash
  )
  expect_false(
    po("torch_callbacks", "history")$phash == po("torch_callbacks", "progress")$phash
  )
})
