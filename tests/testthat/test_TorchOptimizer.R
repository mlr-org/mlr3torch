test_that("Basic checks", {
  torch_opt = TorchOptimizer$new(
    torch_optimizer = optim_ignite_adam,
    label = "Adam",
    packages = "mypackage"
  )
  expect_equal(torch_opt$id, "optim_ignite_adam")
  expect_r6(torch_opt, "TorchOptimizer")
  expect_set_equal(torch_opt$packages, c("mypackage", "torch", "mlr3torch"))
  expect_equal(torch_opt$label, "Adam")
  expect_set_equal(setdiff(torch_opt$param_set$ids(), "param_groups"), setdiff(formalArgs(optim_ignite_adam), c("params")))
  expect_error(torch_opt$generate(), regexp = "could not be loaded: mypackage", fixed = TRUE)

  expect_error(
    TorchOptimizer$new(
      torch_optimizer = optim_sgd,
      param_set = ps(lr = p_uty(), params = p_uty())
    ),
    regexp = "The name 'params' is reserved for the network parameters.", fixed = TRUE
  )

  torch_opt1 = TorchOptimizer$new(
    torch_optimizer = optim_sgd,
    label = "Stochastic Gradient Descent",
    id = "Sgd"
  )

  torch_opt1$param_set$set_values(lr = 0.9191)
  expect_set_equal(torch_opt1$packages, c("torch", "mlr3torch"))
  expect_equal(torch_opt1$label, "Stochastic Gradient Descent")
  expect_equal(torch_opt1$id, "Sgd")
  expect_equal(torch_opt1$param_set$values$lr, 0.9191)

  opt = torch_opt1$generate(nn_linear(1, 1)$parameters)
  expect_class(opt, "torch_optimizer")
  expect_equal(opt$defaults$lr, 0.9191)

  torch_opt2 = TorchOptimizer$new(
    torch_optimizer = optim_sgd,
    param_set = ps(lr = p_uty())
  )
  expect_permutation(torch_opt2$param_set$ids(), c("lr", "param_groups"))
})


test_that("dictionary retrieval works", {
  torch_opt = t_opt("adam", lr = 0.99)
  expect_r6(torch_opt, "TorchOptimizer")
  expect_class(torch_opt$generator, "optim_ignite_adam")
  expect_equal(torch_opt$param_set$values$lr, 0.99)

  descriptors = t_opts(c("adam", "sgd"))
  expect_list(descriptors, types = "TorchOptimizer")
  expect_identical(ids(descriptors), c("adam", "sgd"))

  expect_class(t_opt(), "DictionaryMlr3torchOptimizers")
  expect_class(t_opts(), "DictionaryMlr3torchOptimizers")
})


test_that("dictionary can be converted to a table", {
  tbl = as.data.table(mlr3torch_optimizers)
  expect_data_table(tbl, ncols = 3, key = "key")
  expect_equal(colnames(tbl), c("key", "label", "packages"))

})

test_that("Cloning works", {
  torch_opt1 = t_opt("adam")
  torch_opt2 = torch_opt1$clone(deep = TRUE)
  expect_deep_clone(torch_opt1, torch_opt2)
})

test_that("Printer works", {
  observed = capture.output(print(t_opt("adam")))
  expected = c(
   "<TorchOptimizer:adam> Adaptive Moment Estimation",
   "* Generator: optim_ignite_adam",
   "* Parameters: list()",
   "* Packages: torch,mlr3torch"
  )
  expect_identical(observed, expected)
})

test_that("Converters are correctly implemented", {
  expect_r6(as_torch_optimizer("adam"), "TorchOptimizer")
  torch_opt = as_torch_optimizer(optim_adam)
  expect_r6(torch_opt, "TorchOptimizer")
  expect_equal(torch_opt$id, "optim_adam")
  expect_equal(torch_opt$label, "optim_adam")

  torch_opt1 = as_torch_optimizer(torch_opt, clone = TRUE)
  expect_deep_clone(torch_opt, torch_opt1)

  torch_op2 = as_torch_optimizer(optim_adam, id = "myopt", label = "Custom",
    man = "my_opt", param_set = ps(lr = p_uty(tags = "train"))
  )
  expect_r6(torch_op2, "TorchOptimizer")
  expect_equal(torch_op2$id, "myopt")
  expect_equal(torch_op2$label, "Custom")
  expect_equal(torch_op2$man, "my_opt")
  expect_permutation(torch_op2$param_set$ids(), c("lr", "param_groups"))


  torch_opt3 = as_torch_optimizer(optim_adam)
  expect_equal(torch_opt3$id, "optim_adam")
  expect_equal(torch_opt3$label, "optim_adam")
})

test_that("Parameter test: adamw", {
  torch_opt = t_opt("adamw")
  param_set = torch_opt$param_set
  fn = torch_opt$generator
  res = expect_paramset(param_set, fn, exclude = c("params", "param_groups"))
  expect_paramtest(res)
})

test_that("Parameter test: adam", {
  torch_opt = t_opt("adam")
  param_set = torch_opt$param_set
  fn = torch_opt$generator
  res = expect_paramset(param_set, fn, exclude = c("params", "param_groups"))
  expect_paramtest(res)
})

test_that("Parameter test: sgd", {
  torch_opt = t_opt("sgd")
  param_set = torch_opt$param_set
  # lr is set to `optim_required()`
  fn = torch_opt$generator
  res = expect_paramset(param_set, fn, exclude = c("params", "param_groups", "lr"))
  expect_paramtest(res)
})

test_that("Parameter test: rmsprop", {
  torch_opt = t_opt("rmsprop")
  param_set = torch_opt$param_set
  fn = torch_opt$generator
  res = expect_paramset(param_set, fn, exclude = c("params", "param_groups"))
  expect_paramtest(res)
})

test_that("Parameter test: adagrad", {
  torch_opt = t_opt("adagrad")
  param_set = torch_opt$param_set
  fn = torch_opt$generator
  res = expect_paramset(param_set, fn, exclude = c("params", "param_groups"))
  expect_paramtest(res)
})

test_that("phash works", {
  expect_equal(t_opt("adam", lr = 2)$phash, t_opt("adam", lr = 1)$phash)
  expect_false(t_opt("sgd")$phash == t_opt("adam")$phash)
  expect_false(t_opt("sgd", id = "a")$phash == t_opt("adam", id = "b")$phash)
  expect_false(t_opt("sgd", label = "a")$phash == t_opt("adam", label = "b")$phash)
})

test_that("can train with every optimizer", {
  task = tsk("iris")$filter(1)
  test_optimizer = function(opt_id, ...) {
    opt = t_opt(opt_id, lr = 0.1)
    expect_learner(lrn("classif.mlp", optimizer = opt, batch_size = 1, epochs = 1)$train(task))
  }

  for (opt_id in names(mlr3torch_optimizers$items)) {
    test_optimizer(opt_id)
  }
})

test_that("param groups work", {
  learner = lrn("classif.mlp", neurons = 5, n_layers = 4, epochs = 3, batch_size = 150)
  default_weight_decay = 0.37
  second_weight_decay = 0.11
  learner$param_set$set_values(opt.weight_decay = default_weight_decay)
  learner$param_set$set_values(opt.param_groups = function(params) {
    list(
      list(params = params[1:2]),
      list(params = params[3:length(params)], weight_decay = second_weight_decay)
    )
  })

  task = tsk("iris")$filter(1:10)
  learner$train(task)

  expect_equal(length(learner$model$optimizer$param_groups), 2L)
  expect_equal(learner$model$optimizer$param_groups[[1L]]$weight_decay, default_weight_decay)
  expect_equal(learner$model$optimizer$param_groups[[2L]]$weight_decay, second_weight_decay)

  expect_learner(learner)
})
