test_that("torch_network_predict works", {
  learner = lrn("classif.torch_linear", epochs = 0, batch_size = 1)

  task = tsk("iris")

  net1 = nn_module(
    initialize = function() {
      self$linear1 = nn_linear(1, 3)
      self$linear2 = nn_linear(1, 3)
    },
    forward = function(x1, x2) {
      self$linear1(x1) + self$linear2(x2)
    }
  )()

  net2 = nn_module(
    initialize = function() {
      self$linear1 = nn_linear(1, 3)
      self$linear2 = nn_linear(1, 3)
    },
    forward = function(a1, a2) {
      self$linear1(a1) + self$linear2(a2)
    }
  )()

  ingress1 = list(
    x1 = TorchIngressToken("Sepal.Length", batchgetter_num, c(NA, 1L)),
    x2 = TorchIngressToken("Sepal.Width", batchgetter_num, c(NA, 1L))
  )

  dataset1 = task_dataset(
    task,
    feature_ingress_tokens = ingress1,
    target_batchgetter = crate(function(data, device) {
      torch_tensor(data = as.integer(data[[1]]), dtype = torch_long(), device = device)
    }, .parent = topenv()),
    device = "cpu"
  )

  dataloader1 = dataloader(
    dataset = dataset1,
    batch_size = 3L,
    drop_last = FALSE,
    shuffle = TRUE
  )
  pred = torch_network_predict(net1, dataloader1)
  expect_error(torch_network_predict(net2, dataloader1))


  ingress2 = list(
    x1 = TorchIngressToken("Sepal.Length", batchgetter_num, c(NA, 1L))
  )

  dataset2 = task_dataset(
    task,
    feature_ingress_tokens = ingress2,
    target_batchgetter = crate(function(data, device) {
      torch_tensor(data = as.integer(data[[1]]), dtype = torch_long(), device = device)
    }, .parent = topenv()),
    device = "cpu"
  )

  dataloader2 = dataloader(
    dataset = dataset2,
    batch_size = 3L,
    drop_last = FALSE,
    shuffle = TRUE
  )

  net3 = nn_linear(1, 3)

  weight_before = torch_clone(net3$weight)

  pred = torch_network_predict(net3, dataloader2)

  expect_true(torch_equal(weight_before, net3$weight))

})

test_that("Test roles are respected", {
  task = tsk("iris")
  task$filter(1:20)
  task$set_row_roles(1:10, "test")

  learner = lrn("classif.torch_linear", epochs = 2, batch_size = 1, measures_train = msrs(c("classif.acc")))
  learner$train(task)

  expect_data_table(learner$hist_train, nrows = 2)
  expect_equal(colnames(learner$hist_train), c("epoch", "classif.acc"))

  expect_true(nrow(learner$hist_valid) == 0)
  expect_equal(colnames(learner$hist_valid), "epoch")

  learner = lrn("classif.torch_linear", epochs = 2, batch_size = 1, measures_train = msrs(c("classif.acc")),
    measures_valid = msr("classif.bacc")
  )

  learner$train(task)

  expect_true(nrow(learner$hist_train) == 2)
  expect_true(nrow(learner$hist_valid) == 2)

  learner$hist_valid
})

test_that("train_loop works", {
  learner = lrn("classif.mlp", )

})

test_that("learner_torch_predict works", {


})
