test_that("Basic Checks", {
  torchloss = DescriptorTorchLoss$new(
    torch_loss = nn_cross_entropy_loss,
    label = "Cross Entropy Loss",
    task_types = "classif",
    packages = "mypackage"
  )

  expect_equal(torchloss$id, "nn_cross_entropy_loss")
  expect_r6(torchloss, "DescriptorTorchLoss")
  expect_set_equal(torchloss$packages, c("mypackage", "torch", "mlr3torch"))
  expect_equal(torchloss$label, "Cross Entropy Loss")
  expect_true(torchloss$task_types == "classif")
  expect_set_equal(torchloss$param_set$ids(), setdiff(formalArgs(torch::nn_cross_entropy_loss), "params"))

  expect_error(torchloss$generate(),
    regexp = "The following packages could not be loaded: mypackage", fixed = TRUE
  )

  torchloss1 = DescriptorTorchLoss$new(
    torch_loss = torch::nn_cross_entropy_loss,
    label = "Cross Entropy Loss",
    task_types = "classif"
  )
  torchloss1$param_set$set_values(ignore_index = 123)
  expect_set_equal(torchloss1$packages, c("torch", "mlr3torch"))
  expect_equal(torchloss1$label, "Cross Entropy Loss")

  loss = torchloss1$generate()
  expect_class(loss, c("nn_crossentropy_loss", "nn_module"))
  expect_true(loss$ignore_index == 123)

  expect_error(DescriptorTorchLoss$new(torch::nn_mse_loss, id = "mse", task_types = "regr", param_set = ps(par = p_uty())),
    regexp = "Parameter values with ids 'par' are missing in generator.", fixed = TRUE
  )

  torchloss2 = DescriptorTorchLoss$new(torch::nn_mse_loss, id = "mse", task_types = "regr", param_set = ps(reduction = p_uty()))

  expect_equal(torchloss2$param_set$ids(), "reduction")
  expect_equal(torchloss2$label, "Mse")
  expect_true(torchloss2$task_types == "regr")

})

test_that("dictionary retrieval works", {
  torchloss = t_loss("cross_entropy", ignore_index = 1)
  expect_class(torchloss, "DescriptorTorchLoss")
  expect_class(torchloss$generator, "nn_crossentropy_loss")
  expect_equal(torchloss$param_set$values$ignore_index, 1)

  torchlosses = t_losses(c("cross_entropy", "mse"))
  expect_list(torchlosses, types = "DescriptorTorchLoss", len = 2)
  expect_identical(ids(torchlosses), c("cross_entropy", "mse"))

  expect_class(t_loss(), "DictionaryMlr3torchLosses")
  expect_class(t_losses(), "DictionaryMlr3torchLosses")
})

test_that("dictionary can be converted to table", {
  tbl = as.data.table(mlr3torch_losses)
  expect_data_table(tbl, ncols = 4, key = "key")
  expect_equal(colnames(tbl), c("key", "label", "task_types", "packages"))
})

test_that("Cloning works", {
  torchloss1 = t_loss("cross_entropy")
  torchloss2 = torchloss1$clone(deep = TRUE)
  expect_deep_clone(torchloss1, torchloss2)

  torchloss3 = as_descriptor_torch_loss(torchloss1, clone = TRUE)

  expect_deep_clone(torchloss1, torchloss3)
})

test_that("Printer works", {
  observed = capture.output(print(t_loss("cross_entropy")))

  expected = c(
    "<DescriptorTorchLoss:cross_entropy> Cross Entropy",
    "* Generator: nn_crossentropy_loss",
    "* Parameters: list()",
    "* Packages: torch,mlr3torch",
    "* Task Types: classif"
  )

  expect_identical(observed, expected)
})

test_that("dictionary can be converted to a table", {
  tbl = as.data.table(t_losses())
  expect_data_table(tbl, ncols = 4)
  expect_equal(colnames(tbl), c("key", "label", "task_types", "packages"))
})

test_that("Converters are correctly implemented", {
  expect_r6(as_descriptor_torch_loss("l1"), "DescriptorTorchLoss")
  loss = as_descriptor_torch_loss(torch::nn_l1_loss, task_types = "regr")
  expect_equal(loss$task_types, "regr")
  expect_r6(loss, "DescriptorTorchLoss")
  expect_r6(as_descriptor_torch_loss(t_loss("l1")), "DescriptorTorchLoss")

  expect_error(as_descriptor_torch_loss(nn_l1_loss), "task_types")

  loss1 = as_descriptor_torch_loss(loss, clone = TRUE)
  expect_deep_clone(loss, loss1)

  loss2 = as_descriptor_torch_loss(torch::nn_mse_loss, id = "ce", label = "CE", man = "nn_cross_entropy_loss",
    param_set = ps(reduction = p_uty()), task_types = "regr"
  )
  expect_r6(loss2, "DescriptorTorchLoss")
  expect_equal(loss2$id, "ce")
  expect_equal(loss2$label, "CE")
  expect_equal(loss2$task_types, "regr")
  expect_equal(loss2$man, "nn_cross_entropy_loss")
  expect_equal(loss2$param_set$ids(), "reduction")
})

test_that("Parameter test: mse", {
  loss_mse = t_loss("mse")
  param_set = loss_mse$param_set
  fn = loss_mse$generator
  res = autotest_paramset(param_set, fn)
  expect_equal(res, list())
})

test_that("Parameter test: l1", {
  loss = t_loss("l1")
  param_set = loss$param_set
  fn = loss$generator
  res = autotest_paramset(param_set, fn)
  expect_equal(res, list())
})

test_that("Parameter test: cross_entropy", {
  loss = t_loss("cross_entropy")
  param_set = loss$param_set
  fn = loss$generator
  # ignore_index has param
  res = autotest_paramset(param_set, fn)
  expect_equal(res, list())
})
