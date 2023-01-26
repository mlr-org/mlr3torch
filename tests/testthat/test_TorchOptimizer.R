test_that("TorchOptimizer is correctly initialized", {
  torchopt = TorchOptimizer$new(
    torch_optimizer = torch::optim_sgd,
    label = "Stochastic Gradient Descent",
    packages = "mypackage"
  )

  expect_r6(torchopt, "TorchOptimizer")
  expect_set_equal(torchopt$packages, c("mypackage", "torch", "mlr3torch"))
  expect_equal(torchopt$label, "Stochastic Gradient Descent")

  expect_set_equal(torchopt$param_set$ids(), setdiff(formalArgs(torch::optim_sgd), "params"))


  expect_error(torchopt$get_optimizer(nn_linear(1, 1)$parameters),
    regexp = "The following packages could not be loaded: mypackage", fixed = TRUE
  )

  torchopt = TorchOptimizer$new(
    torch_optimizer = torch::optim_sgd,
    label = "Stochastic Gradient Descent"
  )
  torchopt$param_set$set_values(lr = 0.123)
  expect_set_equal(torchopt$packages, c("torch", "mlr3torch"))
  expect_equal(torchopt$label, "Stochastic Gradient Descent")

  opt = torchopt$get_optimizer(nn_linear(1, 1)$parameters)
  expect_r6(opt, c("optim_sgd", "torch_optimizer"))
  expect_true(opt$param_groups[[1]]$lr == 0.123)

  torchopt = TorchOptimizer$new(
    torch_optimizer = torch::optim_sgd,
    label = "Stochastic Gradient Descent",
    param_set = ps(lr = p_dbl())
  )
  expect_equal(torchopt$param_set$ids(), "lr")
})

test_that("Deep clone of optimizer works", {
  torchopt = t_opt("sgd")

  torchopt1 = torchopt$clone(deep = TRUE)
  torchopt1$param_set$set_values(lr = 0.111)
  expect_true(is.null(torchopt$param_set$values$lr))
  expect_true(torchopt1$param_set$values$lr == 0.111)
})

test_that("Converters are correctly implemented", {
  expect_r6(as_torch_optimizer("sgd"), "TorchOptimizer")
  expect_r6(as_torch_optimizer(torch::optim_sgd), "TorchOptimizer")
  expect_r6(as_torch_optimizer(t_opt("sgd")), "TorchOptimizer")
})

test_that("Dictionary of Torch Optimizers is correctlu specified", {
  walk(mlr3torch_optimizers$keys(), function(key) {
    torchopt = t_opt(key)
    param_set = torchopt$param_set

    expect_true(all(map_lgl(param_set$params$tags, function(tags) tag == "train")))

    observed = torchopt$param_set$default[sort(names(torchopt$param_set$default))]

    # torch marks required parameters with `optim_required()`
    expected = formals(torchopt$optimizer)
    expected = expected[sort(setdiff(names(expected), "params"))]
    required_params = names(expected[map_lgl(expected, function(x) identical(x, str2lang("optim_required()")))])

    for (required_param in required_params) {
      expect_true("required" %in% param_set$params[[required_param]]$tags)
      # required params should not have a default
      expect_true(required_param %nin% observed)
    }

    # required params were already checked
    expected[required_params] = NULL

    # formals stored the expressions
    expected = map(expected, eval)
    expect_true(all.equal(observed, expected))
  })
})
