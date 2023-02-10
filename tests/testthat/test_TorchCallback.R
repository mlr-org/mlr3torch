test_that("Can retrieve predefined callback", {
  x = t_clbk("checkpoint")
  expect_class(x, "TorchCallback")

  cb = t_clbk("checkpoint", freq = 2)
  expect_class(cb, "TorchCallback")
  expect_equal(cb$param_set$values$freq, 2)
})

test_that("TorchCallback basic checks", {
  Cbt1 = R6Class("CallbackTorchTest1")
  expect_error(TorchCallback$new(Cbt1),
    "Callback generator must have public field 'id'.")

  Cbt2 = R6Class("CallbackTorchTest2", public = list(id = "test2"))
  tcb2 = TorchCallback$new(Cbt2)
  expect_identical(Cbt2, tcb2$callback)
  expect_identical(tcb2$id, "test2")
  expect_identical(tcb2$packages, "mlr3torch")
  expect_identical(tcb2$id, tcb2$callback$public_fields$id)

  Cbt3 = R6Class("CallbackTorchTest3", public = list(id = "test3"))
  tcb2 = TorchCallback$new(Cbt2, packages = c("a", "b"))
  expect_set_equal(tcb2$packages, c("mlr3torch", "a", "b"))

  Cbt4 = R6Class("CallbackTorchTest4", public = list(id = "test4", initialize = function(x) NULL))
  tcb41 = TorchCallback$new(Cbt4)
  expect_identical(tcb41$param_set$ids(), "x")
  expect_class(tcb41$param_set$params$x, "ParamUty")

  ps42 = ps(x = p_int())
  tcb42 = TorchCallback$new(Cbt4, param_set = ps42)

  addr1 = data.table::address(ps42)
  addr2 = data.table::address(tcb42$param_set)
  expect_true(addr1 == addr2)

  Cbt5 = R6Class("CallbackTorchTest5", inherit = Cbt4, public = list(id = "test5"))
  tcb5 = TorchCallback$new(Cbt5)
  expect_true(tcb5$id == "test5")
  expect_true(tcb5$param_set$ids() == "x")
})

test_that("Deep clone works", {
  Cbt1 = R6Class("CallbackTorchTest1", public = list(id = "test1"))
  tcb1 = TorchCallback$new(Cbt1)

  tcb2 = tcb1$clone(deep = TRUE)

  # We don't need to copy the class
  expect_true(data.table::address(tcb1$callback) == data.table::address(tcb2$callback))
  expect_true(data.table::address(tcb1$param_set) != data.table::address(tcb2$param_set))

  tcb3 = tcb1$clone(deep = FALSE)

  expect_true(data.table::address(tcb1$callback) == data.table::address(tcb3$callback))
  expect_true(data.table::address(tcb1$param_set) == data.table::address(tcb3$param_set))
})

test_that("S3 converter work as expected", {
  tcb1 = t_clbk("history")
  expect_identical(tcb1, as_torch_callback(tcb1))
  expect_true(data.table::address(tcb1$param_set) != data.table::address(as_torch_callback(tcb1, clone = TRUE)))
  expect_identical(tcb1, as_torch_callback("history"))
  expect_identical(tcb1, as_torch_callback(CallbackTorchHistory))

  Cbt2 = R6Class("CallbackTorchTest1", public = list(id = "test1"))
  ps2 = ps()
  tcb2 = as_torch_callback(Cbt2, param_set = ps2, clone = TRUE)
  expect_true(data.table::address(ps2) != data.table::address(tcb2$param_set))

})
