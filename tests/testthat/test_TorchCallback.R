test_that("Basic checks", {
  Cbt1 = R6Class("CallbackTorchTest1")
  Cbt2 = R6Class("CallbackTorchTest2")
  tcb2 = TorchCallback$new(Cbt2, id = "test2")
  expect_identical(Cbt2, tcb2$generator)
  expect_identical(tcb2$id, "test2")
  expect_identical(tcb2$label, "Test2")
  expect_set_equal(tcb2$packages, c("torch", "mlr3torch"))

  Cbt3 = R6Class("CallbackTorchTest3")
  tcb2 = TorchCallback$new(Cbt2, packages = c("a", "b"))
  expect_set_equal(tcb2$packages, c("torch", "mlr3torch", "a", "b"))

  Cbt4 = R6Class("CallbackTorchTest4", public = list(initialize = function(x) NULL))
  tcb41 = TorchCallback$new(Cbt4)
  expect_identical(tcb41$param_set$ids(), "x")
  expect_class(tcb41$param_set$params$x, "ParamUty")

  ps42 = ps(x = p_int())
  tcb42 = TorchCallback$new(Cbt4, param_set = ps42)

  addr1 = data.table::address(ps42)
  addr2 = data.table::address(tcb42$param_set)
  expect_identical(addr1, addr2)

  Cbt5 = R6Class("CallbackTorchTest5", inherit = Cbt4)
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

test_that("dictionary can be coverted to a table", {
  tbl = as.data.table(mlr3torch_callbacks)

  expect_data_table(tbl, ncols = 3, key = "key")
  expect_equal(colnames(tbl), c("key", "label", "packages"))
})

test_that("torch_callback helper function works", {
  stages = formalArgs(torch_callback)
  stages = stages[grepl("^on_", stages)]
  expect_set_equal(stages, mlr3torch_callback_stages)

  expect_warning(torch_callback(id = "Custom", private = list(
    on_edn = function(ctx) NULL, on_nde = function(ctx) NULL)))

  tcb = torch_callback("Custom",
    on_end = function(ctx) NULL,
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

  expect_class(cbt, "CallbackTorchCustom")
  expect_true(!is.null(get_private(cbt)$on_end))
  expect_true(cbt$a == 1)
  expect_true(get_private(cbt)$b == 2)
})



test_that("S3 converter work as expected", {
  tcb1 = t_clbk("history")
  expect_identical(tcb1, as_torch_callback(tcb1))
  expect_true(data.table::address(tcb1$param_set) != data.table::address(as_torch_callback(tcb1, clone = TRUE)))
  expect_identical(tcb1, as_torch_callback("history"))

  Cbt1 = R6Class("CallbackTorchTest1")
  ps2 = ps()
  tcb2 = as_torch_callback(Cbt1, param_set = ps2)

  expect_equal(tcb2$id, "Cbt1")
  expect_equal(tcb2$label, "Cbt1")
  tcb3 = as_torch_callback(tcb2, clone = TRUE)
  expect_deep_clone(tcb2, tcb3)

  test = R6Class("CallbackTorchTest2", public = list(initialize = function(a) NULL))
  tcb4 = as_torch_callback(test)
  expect_true(tcb4$id == "test")
  expect_true(tcb4$label == "Test")
  expect_equal(tcb4$param_set$ids(), "a")
})


test_that("Cloning works", {
  tcb1 = t_clbk("progress")
  tcb2 = tcb1$clone(deep = TRUE)
  expect_deep_clone(tcb1, tcb2)
})

for (key in mlr3torch_callbacks$keys()) {
  test_that(sprintf("mlr3torch_callbacks: '%s'", key), {
    tcb = t_clbk(key)
    expect_class(tcb, "TorchCallback")
    expect_r6(tcb$param_set, "ParamSet")
    expect_string(tcb$id)
    expect_string(tcb$label)
    expect_man_exists(tcb$man)
  })
}
