test_that("TorchLoss is correctly initialized", {
  torchloss = TorchLoss$new(
    torch_loss = torch::nn_cross_entropy_loss,
    label = "Cross Entropy Loss",
    task_types = "classif",
    packages = "mypackage"
  )

  expect_r6(torchloss, "TorchLoss")
  expect_set_equal(torchloss$packages, c("mypackage", "torch", "mlr3torch"))
  expect_equal(torchloss$label, "Cross Entropy Loss")
  expect_true(torchloss$task_types == "classif")
  expect_set_equal(torchloss$param_set$ids(), setdiff(formalArgs(torch::nn_cross_entropy_loss), "params"))


  expect_error(torchloss$generate(),
    regexp = "The following packages could not be loaded: mypackage", fixed = TRUE
  )

  torchloss = TorchLoss$new(
    torch_loss = torch::nn_cross_entropy_loss,
    label = "Cross Entropy Loss",
    task_types = "classif"
  )
  torchloss$param_set$set_values(ignore_index = 123)
  expect_set_equal(torchloss$packages, c("torch", "mlr3torch"))
  expect_equal(torchloss$label, "Cross Entropy Loss")

  loss = torchloss$generate()
  expect_class(loss, c("nn_crossentropy_loss", "nn_module"))
  expect_true(loss$ignore_index == 123)

  torchloss = TorchLoss$new(
    torch_loss = torch::nn_mse_loss,
    label = "MSE Loss",
    task_types = "regr",
    param_set = ps(par = p_uty())
  )
  expect_equal(torchloss$param_set$ids(), "par")
  expect_true(torchloss$label == "MSE Loss", "par")
  expect_true(torchloss$task_types == "regr")

})

test_that("Deep clone of TorchLoss works", {
  torchloss = t_opt("sgd")

  torchloss1 = torchloss$clone(deep = TRUE)
  torchloss1$param_set$set_values(lr = 0.111)
  expect_true(is.null(torchloss$param_set$values$lr))
  expect_true(torchloss1$param_set$values$lr == 0.111)
})

test_that("Converters are correctly implemented", {
  expect_r6(as_torch_loss("l1"), "TorchLoss")
  loss = as_torch_loss(torch::nn_l1_loss)
  expect_set_equal(loss$task_types, mlr_reflections$task_types$type)
  expect_r6(loss, "TorchLoss")
  expect_r6(as_torch_loss(t_loss("l1")), "TorchLoss")
})

test_that("Default values of torch losses are correctly specified", {
  walk(mlr3torch_losses$keys(), function(key) {
    torchloss = t_loss(key)
    param_set = torchloss$param_set
    observed = torchloss$param_set$default[sort(names(torchloss$param_set$default))]

    expect_true(all(map_lgl(param_set$params$tags, function(tags) tag == "train")))

    # torch marks required parameters with `loss_required()`
    expected = formals(torchloss$generator)
    expected = expected[sort(names(expected))]
    required_params = names(expected[map_lgl(expected, function(x) identical(x, str2lang("loss_required()")))])

    for (required_param in required_params) {
      expect_true("required" %in% param_set$params[[required_param]]$tags)
      # required params should not have a default
      expect_true(required_param %nin% observed)
    }

    # required params were already checked
    expected[required_params] = NULL

    # formals stored the expressions
    expected = map(expected, eval)
    # expect_true(all.equal(observed, expected))
  })
})
