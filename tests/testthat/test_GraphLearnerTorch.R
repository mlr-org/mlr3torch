# `as_learner_torch()` runs the graph as it is, so the graph has to configure the whole training:
# the model operator, which holds its parameters, and the loss and the optimizer
torch_learner = function(graph, task_type = "classif", id = NULL, ...) {
  has = function(class) some(graph$pipeops, function(po) test_class(po, class))
  if (!has("PipeOpTorchLoss")) {
    graph = graph %>>% po("torch_loss", if (task_type == "classif") "cross_entropy" else "mse")
  }
  if (!has("PipeOpTorchOptimizer")) {
    graph = graph %>>% po("torch_optimizer", "adam")
  }
  as_learner_torch(graph %>>% po(paste0("torch_model_", task_type), ...), id = id)
}

simple_graph = function() {
  po("torch_ingress_num") %>>%
    nn("linear", out_features = 5) %>>%
    nn("relu") %>>%
    nn("head") %>>%
    po("torch_loss", "cross_entropy") %>>%
    po("torch_optimizer", "adam", lr = 0.1)
}

test_that("classification: train and predict", {
  task = tsk("iris")
  learner = torch_learner(simple_graph(), epochs = 1, batch_size = 32)

  expect_class(learner, "GraphLearner")
  expect_equal(learner$id, "classif.graph")
  expect_equal(learner$task_type, "classif")

  learner$train(task)
  expect_class(learner$network, c("nn_graph", "nn_module"))
  expect_class(learner$base_learner(), "LearnerTorchModel")

  pred = learner$predict(task)
  expect_class(pred, "PredictionClassif")
  expect_equal(pred$row_ids, task$row_ids)
})
test_that("the network and the dataset match the ingress of the graph", {
  task = tsk("iris")
  learner = torch_learner(simple_graph(), epochs = 1, batch_size = 32)
  learner$train(task)

  expect_equal(names(learner$network$shapes_in), "torch_ingress_num.input")
  ds = learner$dataset(task)
  batch = ds$.getbatch(1:2)
  expect_equal(names(batch$x), "torch_ingress_num.input")
  out = with_no_grad(invoke(learner$network, .args = batch$x))
  expect_equal(out$shape, c(2, 3))
})

test_that("hyperparameters of the graph are exposed and used", {
  learner = torch_learner(simple_graph(), epochs = 1, batch_size = 32)

  expect_subset(c("linear.out_features", "linear.bias", "head.bias"), learner$param_set$ids())
  expect_equal(learner$param_set$values$linear.out_features, 5)
  # the parameters of the model operator keep its id as prefix
  expect_equal(learner$param_set$values$torch_model_classif.epochs, 1)
  expect_equal(learner$param_set$values$torch_optimizer.lr, 0.1)

  learner$param_set$set_values(linear.out_features = 7)
  learner$train(tsk("iris"))
  expect_equal(learner$network$graph$pipeops$linear$module$out_features, 7)
})

test_that("loss, optimizer and callbacks are read from the graph", {
  graph = simple_graph() %>>% po("torch_callbacks", t_clbk("checkpoint", freq = 2))
  learner = torch_learner(graph, epochs = 1, batch_size = 32)

  expect_equal(learner$loss$id, "cross_entropy")
  expect_equal(learner$optimizer$id, "adam")
  expect_equal(ids(learner$callbacks), "checkpoint")
  expect_equal(learner$param_set$values$torch_callbacks.checkpoint.freq, 2)
})

test_that("the graph configures the training, the id is taken from the argument", {
  path = tempfile()
  graph = po("torch_ingress_num") %>>% nn("head") %>>%
    po("torch_loss", t_loss("cross_entropy", reduction = "sum")) %>>%
    po("torch_optimizer", t_opt("sgd", lr = 0.5)) %>>%
    po("torch_callbacks", t_clbk("checkpoint", freq = 1, path = path))
  learner = torch_learner(graph, id = "my_learner", epochs = 1, batch_size = 32)

  expect_equal(learner$id, "my_learner")
  expect_equal(learner$optimizer$id, "sgd")
  expect_equal(learner$param_set$values$torch_optimizer.lr, 0.5)
  expect_equal(learner$param_set$values$torch_loss.reduction, "sum")
  expect_equal(ids(learner$callbacks), "checkpoint")
  expect_equal(learner$param_set$values$torch_callbacks.checkpoint.freq, 1)

  learner$train(tsk("iris"))
  expect_equal(learner$base_learner()$model$callbacks$checkpoint$path, path)
})

test_that("loss, optimizer and callbacks are read-only, the graph configures them", {
  learner = torch_learner(simple_graph() %>>% po("torch_callbacks", t_clbk("history")),
    epochs = 1, batch_size = 32)

  expect_error({learner$loss = t_loss("cross_entropy")}, "read-only")
  expect_error({learner$optimizer = t_opt("sgd")}, "read-only")
  expect_error({learner$callbacks = t_clbk("history")}, "read-only")

  # they are configured through the operators of the graph and their parameters
  learner$param_set$set_values(torch_loss.reduction = "sum", torch_optimizer.lr = 0.5)
  expect_equal(learner$loss$id, "cross_entropy")
  expect_equal(learner$optimizer$id, "adam")
  expect_equal(ids(learner$callbacks), "history")

  learner$train(tsk("iris"))
  expect_data_table(learner$base_learner()$model$callbacks$history)
})

test_that("PipeOp and GraphLearner methods", {
  learner = torch_learner(po("torch_ingress_num") %>>% nn("head"), task_type = "regr",
    epochs = 1, batch_size = 16)
  expect_class(learner, "GraphLearner")
  learner$train(tsk("mtcars"))
  expect_class(learner$network, "nn_graph")

  glrn = as_learner(simple_graph() %>>% po("torch_model_classif", epochs = 1, batch_size = 32))
  learner2 = as_learner_torch(glrn)
  expect_class(learner2, "GraphLearner")
  expect_equal(learner2$task_type, "classif")
  expect_equal(learner2$id, glrn$id)
  expect_equal(learner2$param_set$values$torch_model_classif.epochs, 1)
})

two_ingress_graph = function() {
  gunion(list(
    po("select", id = "select_sepal", selector = selector_grep("^Sepal")) %>>%
      po("torch_ingress_num", id = "sepal"),
    po("select", id = "select_petal", selector = selector_grep("^Petal")) %>>%
      po("torch_ingress_num", id = "petal")
  )) %>>% nn("merge_cat") %>>% nn("head")
}

test_that("multiple ingress operators", {
  task = tsk("iris")
  learner = torch_learner(two_ingress_graph(), epochs = 1, batch_size = 32)

  learner$train(task)
  expect_set_equal(names(learner$network$shapes_in), c("sepal.input", "petal.input"))
  expect_set_equal(names(learner$dataset(task)$.getbatch(1)$x), c("sepal.input", "petal.input"))
  expect_class(learner$predict(task), "PredictionClassif")
})
batch_mean = function(ds, n) {
  as.numeric(ds$.getbatch(seq_len(n))$x[[1L]]$mean())
}

test_that("the stages of a preprocessing operator are respected", {
  task = tsk("lazy_iris")
  # doubling the features is easy to recognize in the tensors and does not change their shape
  po_augment = po("preproc_torch", id = "double",
    fn = crate(function(x) x * 2), stages = "train")
  graph = po_augment %>>% po("torch_ingress_ltnsr", shape = c(NA, 4)) %>>% nn("head")
  learner = torch_learner(graph, epochs = 1, batch_size = 32)
  learner$train(task)

  raw = batch_mean(torch_learner(po("torch_ingress_ltnsr") %>>% nn("head"),
    epochs = 1, batch_size = 32)$dataset(task), 10L)
  # `stages = "train"` means the transformation belongs to training only
  expect_equal(batch_mean(learner$dataset(task, stage = "train"), 10L), 2 * raw)
  expect_equal(batch_mean(learner$dataset(task), 10L), raw)
})
