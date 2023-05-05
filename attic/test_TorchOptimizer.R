test_that("Dictionary retrieval works", {
  obj = t_opt("adam", lr = 0.991)
  expect_equal(obj$param_set$values$lr, 0.991)
  expect_class(obj, "TorchOptimizer")
  expect_list(t_opts(c("adam", "adagrad")), types = "TorchOptimizer", len = 2)
  expect_class(t_opt(), "DictionaryMlr3torchOptimizers")
  expect_class(t_opts(), "DictionaryMlr3torchOptimizers")
})

test_that("Dictionary can be coverted to table", {
  tbl = as.data.table(mlr3torch_optimizers)
  expect_data_table(tbl, key = "key", ncols = 3)
  expect_equal(colnames(tbl), c("key", "label", "packages"))
})

test_that("Basic checks", {
  torchopt = TorchOptimizer$new(
    torch_optimizer = torch::optim_sgd,
    id = "sgd",
    label = "Stochastic Gradient Descent",
    packages = "mypackage"
  )
  torchopt$param_set$set_values(lr = 0.01)

  expect_r6(torchopt, "TorchOptimizer")
  expect_set_equal(torchopt$packages, c("mypackage", "torch", "mlr3torch"))
  expect_equal(torchopt$label, "Stochastic Gradient Descent")

  expect_set_equal(torchopt$param_set$ids(), setdiff(formalArgs(torch::optim_sgd), "params"))

  expect_error(torchopt$generate(nn_linear(1, 1)$parameters),
    regexp = "The following packages could not be loaded: mypackage", fixed = TRUE
  )

  torchopt1 = TorchOptimizer$new(
    torch_optimizer = torch::optim_sgd,
    id = "sgd",
    label = "Stochastic Gradient Descent"
  )
  torchopt1$param_set$set_values(lr = 0.123)
  expect_set_equal(torchopt1$packages, c("torch", "mlr3torch"))
  expect_set_equal(torchopt1$param_set$ids(), setdiff(formalArgs(torch::optim_sgd), "params"))
  expect_equal(torchopt1$label, "Stochastic Gradient Descent")

  opt = torchopt1$generate(nn_linear(1, 1)$parameters)
  expect_r6(opt, c("optim_sgd", "torch_optimizer"))
  expect_true(opt$param_groups[[1]]$lr == 0.123)

  torchopt2 = TorchOptimizer$new(
    torch_optimizer = torch::optim_sgd,
    label = "Stochastic Gradient Descent",
    id = "sgd",
    param_set = ps(lr = p_dbl())
  )
  expect_equal(torchopt2$param_set$ids(), "lr")

  expect_error(TorchOptimizer$new(optim_sgd, param_set = ps(params = p_uty())),
    regexp = "The name 'params' is reserved for the network parameters.",
    fixed = TRUE
  )

  torchopt3 = TorchOptimizer$new(torch_optimizer = torch::optim_sgd, id = "sgd")
  expect_true(torchopt3$label == "Sgd")
})

test_that("Printer", {
  obj = t_opt("sgd", lr = 0.1)
  repr = capture.output(obj)
  expected = c(
    "<TorchOptimizer:sgd> Stochastic Gradient Descent",
    "* Generator: optim_sgd",
    "* Parameters: lr=0.1",
    "* Packages: torch,mlr3torch"
  )
  expect_identical(repr, expected)
})


test_that("Cloning works", {
  torchopt1 = t_opt("sgd")
  torchopt2 = torchopt1$clone(deep = TRUE)
  expect_deep_clone(torchopt1, torchopt2)
})


test_that("Converters are correctly implemented", {
  expect_r6(as_torch_optimizer("sgd"), "TorchOptimizer")
  expect_r6(as_torch_optimizer(torch::optim_sgd), "TorchOptimizer")
  obj = t_opt("sgd")
  expect_r6(as_torch_optimizer(obj), "TorchOptimizer")

  obj1 = as_torch_optimizer(obj, clone = TRUE)
  expect_deep_clone(obj, obj1)
})

for (key in mlr3torch_optimizers$keys()) {
  test_that(sprintf("mlr3torch_optimizers: '%s'", key), {
    torchopt = try(t_opt(key), silent = TRUE)
    expect_class(torchopt, "TorchOptimizer")
    param_set = torchopt$param_set
    expect_class(param_set, "ParamSet", label = key)
    expect_string(torchopt$label, label = key)
    expect_string(torchopt$id, label = key)
    expect_identical(key, torchopt$id, label = key)

    expect_man_exists(torchopt$man)
    expect_true(all(map_lgl(param_set$params$tags, function(tags) tag == "train")), label = key)

    observed = torchopt$param_set$default[sort(names(torchopt$param_set$default))]

    # torch marks required parameters with `optim_required()`
    expected = formals(torchopt$generator)
    expected = expected[sort(setdiff(names(expected), "params"))]
    required_params = names(expected[map_lgl(expected, function(x) identical(x, str2lang("optim_required()")))])

    for (required_param in required_params) {
      expect_true("required" %in% param_set$params[[required_param]]$tags, label = key)
      # required params should not have a default
      expect_true(required_param %nin% observed, label = key)
    }

    # required params were already checked
    expected[required_params] = NULL

    # formals stored the expressions
    expected = map(expected, eval)
    expect_equal(observed, expected, label = key)
  })
}
