test_that("LearnerTorchModel works", {
  # autotest not possible because network is bound to task
  task = tsk("iris")
  learner = LearnerTorchModel$new(
    task_type = "classif",
    network = testmodule_linear(task),
    packages = "data.table"
  )
  learner$ingress_tokens = list(x = TorchIngressToken(task$feature_names, batchgetter_num, c(NA, 4L)))


  expect_deep_clone_mlr3torch(
    learner, learner$clone(deep = TRUE)
  )

  learner$param_set$set_values(device = "cpu", epochs = 0, batch_size = 3)

  expect_learner(learner)

  expect_deep_clone_mlr3torch(
    learner, learner$clone(deep = TRUE)
  )
  learner$train(task)
  pred = learner$predict(task)
  expect_class(pred, "PredictionClassif")

  expect_error(learner$train(task), "No network stored")

  expect_set_equal(learner$packages, c("data.table", "mlr3", "mlr3torch", "torch"))
  expect_set_equal(learner$predict_types, names(mlr_reflections$learner_predict_types$classif))
})

test_that("cannot clone trained LearnerTorchModel", {
  # this is impossible, because a LearnerTorchModel is initialized with a network that is then trained
  # Once the learner is trained, the initial state of the network cannot be accessed anymore
  task = tsk("iris")
  learner = LearnerTorchModel$new(
    task_type = "classif",
    network = testmodule_linear(task),
    ingress_tokens = list(x = TorchIngressToken(task$feature_names, batchgetter_num, c(NA, 4L))),
    packages = "data.table",
  )
  learner$param_set$set_values(
    epochs = 0, batch_size = 50
  )

  learner$train(task)
  expect_error(learner$clone(deep = TRUE), "for untrained")
})

test_that("marshaling works for graph learner", {
  graph = po("torch_ingress_num") %>>%
    po("nn_linear", out_features = 20) %>>%
    po("nn_relu") %>>%
    po("nn_head") %>>%
    po("torch_loss", loss = t_loss("cross_entropy")) %>>%
    po("torch_optimizer", optimizer = t_opt("adam", lr = 0.1)) %>>%
    po("torch_callbacks", callbacks = t_clbk("history")) %>>%
    po("torch_model_classif", batch_size = 50, epochs = 1, device = "cpu")

  learner = as_learner(graph)
  learner$id = "graph_mlp"
  task = tsk("iris")
  learner$train(task)
  learner$marshal()
  learner$unmarshal()
  expect_class(learner$predict(task), "Prediction")
})

test_that("LearnerTorchModel and marshaling", {
  # there used to be a marshaling bug resulting from the fact that composed network
  # is stored in the learner (not part of the model)
  task = tsk("iris")
    learner = LearnerTorchModel$new(
    task_type = "classif",
    network = testmodule_linear(task),
    ingress_tokens = list(x = TorchIngressToken(task$feature_names, batchgetter_num, c(NA, 4L))),
    packages = "data.table",
  )
  learner$encapsulate("callr", lrn("classif.featureless"))
  learner$param_set$set_values(
    batch_size = 50,
    epochs = 1
  )
  learner$train(task)
  expect_class(learner$model, "learner_torch_model")
})

test_that("training does not change the hash of a LearnerTorchModel", {
  # `.network()` consumes `.network_stored`, so a `$phash` built from that field changed as soon as
  # the learner was trained. The identity is recorded once when the network is stored instead.
  tokens = list(x = TorchIngressToken(c("Sepal.Length", "Sepal.Width"), batchgetter_num, c(NA, 2)))
  network = nn_linear(2, 3)
  mk = function(net = network) {
    lrn("classif.torch_model", network = net, ingress_tokens = tokens, epochs = 1, batch_size = 16,
      device = "cpu")
  }

  learner = mk()
  before = learner$hash
  learner$train(tsk("iris"))
  expect_equal(before, learner$hash)

  expect_equal(mk()$hash, mk()$hash)
  expect_equal(mk()$hash, mk()$clone(deep = TRUE)$hash)
  expect_false(identical(mk()$hash, mk(nn_linear(2, 3))$hash))
})

test_that("training does not change the hash of a graph-built learner", {
  graph = po("torch_ingress_num") %>>% nn("head") %>>%
    po("torch_loss", t_loss("cross_entropy")) %>>% po("torch_optimizer", t_opt("adam")) %>>%
    po("torch_model_classif", batch_size = 16, epochs = 1, device = "cpu")
  learner = as_learner(graph)
  before = learner$hash
  learner$train(tsk("iris"))
  expect_equal(before, learner$hash)
})

test_that("a graph-built learner takes its identity from the graph", {
  # `PipeOpTorchModel$.train()` assigns the unserialized module to skip the serialization round-trip,
  # so there is nothing to record the identity from except the graph the network was built from.
  # Before, the hash was taken from the module object, which made two learners built from the same
  # graph differ and -- since a bare `nn_module` digests to its formals and body, which every module
  # shares -- two learners built from *different* graphs agree.
  mk = function(out_features) {
    graph = po("torch_ingress_num") %>>% nn("linear", out_features = out_features) %>>% nn("head") %>>%
      po("torch_loss", t_loss("cross_entropy")) %>>% po("torch_optimizer", t_opt("adam"))
    md = as_graph(graph)$train(tsk("iris"))[[1L]]
    po_model = po("torch_model_classif", batch_size = 16, epochs = 1, device = "cpu")
    po_model$train(list(md))
    po_model$learner$phash
  }

  expect_equal(mk(3), mk(3))
  expect_false(identical(mk(3), mk(7)))
})
