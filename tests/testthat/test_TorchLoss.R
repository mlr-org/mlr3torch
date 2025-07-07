test_that("Basic Checks", {
  torchloss = TorchLoss$new(
    torch_loss = nn_cross_entropy_loss,
    label = "Cross Entropy Loss",
    task_types = "classif",
    packages = "mypackage"
  )

  expect_equal(torchloss$id, "nn_cross_entropy_loss")
  expect_r6(torchloss, "TorchLoss")
  expect_set_equal(torchloss$packages, c("mypackage", "torch", "mlr3torch"))
  expect_equal(torchloss$label, "Cross Entropy Loss")
  expect_true(torchloss$task_types == "classif")
  expect_set_equal(torchloss$param_set$ids(), formalArgs(torch::nn_cross_entropy_loss))

  expect_error(torchloss$generate(tsk("iris")),
    regexp = "The following packages could not be loaded: mypackage"
  )

  torchloss1 = TorchLoss$new(
    torch_loss = torch::nn_cross_entropy_loss,
    label = "Cross Entropy Loss",
    task_types = "classif",
    id = "xe"
  )
  torchloss1$param_set$set_values(ignore_index = 123)
  expect_set_equal(torchloss1$packages, c("torch", "mlr3torch"))
  expect_equal(torchloss1$label, "Cross Entropy Loss")
  expect_equal(torchloss1$id, "xe")

  loss = torchloss1$generate()
  expect_class(loss, c("nn_cross_entropy_loss", "nn_module"))
  expect_true(loss$ignore_index == 123)

  torchloss2 = TorchLoss$new(torch::nn_mse_loss, id = "mse", task_types = "regr", param_set = ps(reduction = p_uty()))

  expect_equal(torchloss2$param_set$ids(), "reduction")
  expect_equal(torchloss2$label, "mse")
  expect_true(torchloss2$task_types == "regr")
})

test_that("dictionary retrieval works", {
  torchloss = t_loss("cross_entropy", ignore_index = 1)
  expect_class(torchloss, "TorchLoss")
  expect_class(torchloss$generator, "function")
  expect_equal(torchloss$param_set$values$ignore_index, 1)

  torchlosses = t_losses(c("cross_entropy", "mse"))
  expect_list(torchlosses, types = "TorchLoss", len = 2)
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
})

test_that("Printer works", {
  observed = capture.output(print(t_loss("cross_entropy")))

  expected = c(
    "<TorchLoss:cross_entropy> Cross Entropy",
    "* Generator: function",
    "* Parameters: list()",
    "* Packages: torch,mlr3torch",
    "* Task Types: classif"
  )

  expect_identical(observed, expected)
})


test_that("Converters are correctly implemented", {
  expect_permutation(
    as_torch_loss(torch::nn_mse_loss)$task_types,
    c("regr", "classif")
  )
  expect_r6(as_torch_loss("l1"), "TorchLoss")
  loss = as_torch_loss(torch::nn_l1_loss, task_types = "regr")
  expect_equal(loss$task_types, "regr")
  expect_r6(loss, "TorchLoss")
  expect_r6(as_torch_loss(t_loss("l1")), "TorchLoss")

  loss1 = as_torch_loss(loss, clone = TRUE)
  expect_deep_clone(loss, loss1)

  loss2 = as_torch_loss(torch::nn_mse_loss, id = "ce", label = "CE", man = "nn_cross_entropy_loss",
    param_set = ps(reduction = p_uty()), task_types = "regr"
  )
  expect_r6(loss2, "TorchLoss")
  expect_equal(loss2$id, "ce")
  expect_equal(loss2$label, "CE")
  expect_equal(loss2$task_types, "regr")
  expect_equal(loss2$man, "nn_cross_entropy_loss")
  expect_equal(loss2$param_set$ids(), "reduction")

  loss3 = as_torch_loss(nn_mse_loss, task_types = "regr")
  expect_equal(loss3$id, "nn_mse_loss")
  expect_equal(loss3$label, "nn_mse_loss")
})

test_that("Parameter test: mse", {
  loss_mse = t_loss("mse")
  param_set = loss_mse$param_set
  fn = loss_mse$generator
  res = expect_paramset(param_set, fn)
  expect_paramtest(res)
})

test_that("Parameter test: l1", {
  loss = t_loss("l1")
  param_set = loss$param_set
  fn = loss$generator
  res = expect_paramset(param_set, fn)
  expect_paramtest(res)
})

test_that("phash works", {
  expect_equal(t_loss("mse", reduction = "mean")$phash, t_loss("mse", reduction = "sum")$phash)
  expect_false(t_loss("mse")$phash == t_loss("l1")$phash)
  expect_false(t_loss("mse", id = "a")$phash == t_loss("mse", id = "b")$phash)
  expect_false(t_loss("mse", label = "a")$phash == t_loss("mse", label = "b")$phash)
  expect_false(t_loss("mse", task_types = "regr")$phash == t_loss("mse", task_types = "classif")$phash)
})

test_that("all classif losses can be used to train", {
  task = tsk("iris")$filter(1)
  classif_losses = as.data.table(mlr3torch_losses)[
    map_lgl(get("task_types"), function(x) "classif" %in% x), "key"][[1L]]
  for (loss_id in classif_losses) {
    expect_learner(lrn("classif.mlp", loss = t_loss(loss_id), epochs = 1L, batch_size = 1L)$train(task))
  }
})

test_that("all regr losses can be used to train", {
  task = tsk("mtcars")$filter(1)
  regr_losses = as.data.table(mlr3torch_losses)[
    map_lgl(get("task_types"), function(x) "regr" %in% x), "key"][[1L]]
  for (loss_id in regr_losses) {
    expect_learner(lrn("regr.mlp", loss = t_loss(loss_id), epochs = 1L, batch_size = 1L)$train(task))
  }
})

test_that("cross entropy", {
  loss_binary = t_loss("cross_entropy")
  loss_multi = t_loss("cross_entropy")

  tsk_binary = tsk("iris")$filter(1:100)$droplevels()
  tsk_multi = tsk("iris")

  expect_class(fn <<- loss_binary$generate(tsk_binary), "nn_bce_with_logits_loss")
  expect_equal(fn, nn_bce_with_logits_loss())

  loss_binary$param_set$set_values(class_weight = torch_tensor(1))
  expect_class(fn <<- loss_binary$generate(tsk_binary), "nn_bce_with_logits_loss")
  expect_equal(fn, nn_bce_with_logits_loss(pos_weight = 1))

  loss_binary$param_set$set_values(ignore_index = 1)
  expect_error(loss_binary$generate(tsk_binary), "ignore_index is not supported for binary cross entropy loss")
  loss_binary$param_set$set_values(ignore_index = NULL, reduction = "sum")
  expect_class(loss_binary$generate(tsk_binary), "nn_bce_with_logits_loss")

  l_multi = loss_multi$generate(tsk_multi)
  expect_class(fn <- l_multi, "nn_cross_entropy_loss")
  expect_equal(fn, nn_cross_entropy_loss())

  loss_multi$param_set$set_values(class_weight = torch_tensor(1))
  expect_class(fn <<- loss_multi$generate(tsk_multi), "nn_cross_entropy_loss")
  expect_equal(fn, nn_cross_entropy_loss(weight = 1))

  loss_multi$param_set$set_values(ignore_index = 3)
  expect_class(fn <- loss_multi$generate(tsk_multi), "nn_cross_entropy_loss")
  expect_equal(fn, nn_cross_entropy_loss(ignore_index = 3))

  loss_multi$param_set$set_values(ignore_index = NULL, reduction = "sum")
  expect_class(fn <- loss_multi$generate(tsk_multi), "nn_cross_entropy_loss")
  expect_equal(fn, nn_cross_entropy_loss(reduction = "sum"))
})
