test_that("Basic checks", {
  Cbt1 = R6Class("CallbackSetTest1")
  Cbt2 = R6Class("CallbackSetTest2")
  tcb2 = TorchCallback$new(Cbt2, id = "test2")
  expect_identical(Cbt2, tcb2$generator)
  expect_identical(tcb2$id, "test2")
  expect_identical(tcb2$label, "test2")
  expect_set_equal(tcb2$packages, c("torch", "mlr3torch"))

  Cbt3 = R6Class("CallbackSetTest3")
  tcb2 = TorchCallback$new(Cbt2, packages = c("a", "b"))
  expect_set_equal(tcb2$packages, c("torch", "mlr3torch", "a", "b"))

  Cbt4 = R6Class("CallbackSetTest4", public = list(initialize = function(x) NULL))
  tcb41 = TorchCallback$new(Cbt4)
  expect_identical(tcb41$param_set$ids(), "x")
  expect_equal(tcb41$param_set$params[list("x"), "cls", on = "id"][[1L]], "ParamUty")

  ps42 = ps(x = p_int())
  tcb42 = TorchCallback$new(Cbt4, param_set = ps42)

  addr1 = data.table::address(ps42)
  addr2 = data.table::address(tcb42$param_set)
  expect_identical(addr1, addr2)

  Cbt5 = R6Class("CallbackSetTest5", inherit = Cbt4)
  tcb5 = TorchCallback$new(Cbt5)
  expect_true(tcb5$param_set$ids() == "x")
})

test_that("Can retrieve predefined callback", {
  x = t_clbk("checkpoint")
  expect_class(x, "TorchCallback")

  cb = t_clbk("checkpoint", freq = 2)
  expect_class(cb, "TorchCallback")
  expect_equal(cb$param_set$values$freq, 2)

  cbs = t_clbks(c("checkpoint", "progress"))
  expect_list(cbs, types = "TorchCallback", len = 2L)
  expect_identical(names(cbs), c("checkpoint", "progress"))

  expect_class(t_clbk(), "DictionaryMlr3torchCallbacks")
  expect_class(t_clbks(), "DictionaryMlr3torchCallbacks")
})

test_that("dictionary can be converted to a table", {
  tbl = as.data.table(mlr3torch_callbacks)

  expect_data_table(tbl, ncols = 3, key = "key")
  expect_equal(colnames(tbl), c("key", "label", "packages"))
})

test_that("torch_callback helper function works", {
  stages = formalArgs(torch_callback)
  stages = stages[grepl("^on_", stages)]
  expect_set_equal(stages, mlr_reflections$torch$callback_stages)

  expect_warning(torch_callback(id = "Custom", public = list(
    on_edn = function() NULL, on_nde = function() NULL)))

  tcb = torch_callback("Custom",
    on_end = function() NULL,
    public = list(
      a = 1
    ),
    private = list(
      b = 2
    ),
    packages = "utils"
  )


  expect_class(tcb, "TorchCallback")
  expect_class(tcb$generator, "R6ClassGenerator")
  expect_true("utils" %in% tcb$packages)

  cbt = tcb$generate()

  expect_class(cbt, "CallbackSetCustom")
  expect_true(!is.null(cbt$on_end))
  expect_true(cbt$a == 1)
  expect_true(get_private(cbt)$b == 2)
})



test_that("S3 converter work as expected", {
  tcb1 = t_clbk("history")
  expect_identical(tcb1, as_torch_callback(tcb1))
  expect_true(data.table::address(tcb1$param_set) != data.table::address(as_torch_callback(tcb1, clone = TRUE)))
  expect_identical(tcb1, as_torch_callback("history"))

  Cbt1 = R6Class("CallbackSetTest1")
  ps2 = ps()
  tcb2 = as_torch_callback(Cbt1, param_set = ps2)

  expect_equal(tcb2$id, "Cbt1")
  expect_equal(tcb2$label, "Cbt1")
  tcb3 = as_torch_callback(tcb2, clone = TRUE)
  expect_deep_clone(tcb2, tcb3)

  test = R6Class("CallbackSetTest2", public = list(initialize = function(a) NULL))
  tcb4 = as_torch_callback(test)
  expect_true(tcb4$id == "test")
  expect_true(tcb4$label == "test")
  expect_equal(tcb4$param_set$ids(), "a")
})


test_that("Cloning works", {
  tcb1 = t_clbk("progress")
  tcb2 = tcb1$clone(deep = TRUE)
  expect_deep_clone(tcb1, tcb2)
})
