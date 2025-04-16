test_that("LearnerTorchMLP works", {

  verify_network = function(learner) {
    tab = table(map_chr(learner$network$children, function(x) class(x)[[1L]]))
    act = class(learner$param_set$values$activation)[[1L]]

    l = length(learner$param_set$values$neurons)

    expect_true(tab["nn_linear"] == l + 1)
    if (l > 0) {
      expect_true(tab[act] == l)
      expect_true(tab["nn_dropout"] == l)
    } else {
      # only one nn linear
      expect_true(length(tab) == 1)
    }
  }

  learner = lrn("classif.mlp",
    neurons = rep(13, 2),
    p = 0.111,
    batch_size = 16L,
    epochs = 0,
    optimizer = "adagrad",
    activation = nn_softshrink,
    activation_args = list(lambd = 0.25)
  )
  task = tsk("iris")
  learner$train(task)

  expect_true(nrow(learner$network$children[[1L]]$weight) == 13L)
  expect_true(!is.null(learner$network$children[[1L]]$bias))
  expect_true(learner$network$children[[2L]]$lambd == 0.25)

  expect_learner(learner)
  expect_class(learner$network, c("nn_sequential", "nn_module"))

  verify_network(learner)

  learner$param_set$set_values(neurons = integer(0))

  learner$train(task)
  verify_network(learner)
})

test_that("works for lazy tensor", {
  learner = lrn("classif.mlp", epochs = 100, batch_size = 150)
  task_lazy = tsk("lazy_iris")
  lt = learner$train(task_lazy)
  expect_class(lt, "Learner")
  pred = learner$predict(task_lazy)
  expect_class(pred, "Prediction")
})

test_that("neurons and n_layers", {
  l1 = lrn("classif.mlp", batch_size = 32, epochs = 0L)
  l2 = l1$clone(deep = TRUE)
  task = tsk("iris")
  l1$param_set$set_values(neurons = c(10, 10))
  l2$param_set$set_values(neurons = 10, n_layers = 2)
  l1$train(task)
  l2$train(task)
  expect_equal(l1$network$parameters[[1]]$shape, l2$network$parameters[[1]]$shape)
  expect_equal(l1$network$parameters[[3]]$shape, l2$network$parameters[[3]]$shape)
  expect_equal(l1$network$parameters[[5]]$shape, l2$network$parameters[[5]]$shape)
  expect_equal(l1$network$parameters[[1]]$shape, c(10, 4))
  expect_equal(l1$network$parameters[[3]]$shape, c(10, 10))
  expect_equal(l1$network$parameters[[5]]$shape, c(3, 10))

  l1$param_set$set_values(n_layers = 2)
  expect_error(l1$train(task), "Can only supply")
})

test_that("task types", {
  expect_learner_torch(lrn("classif.mlp", neurons = 10, batch_size = 16, epochs = 0), tsk("iris"))
  expect_learner_torch(lrn("classif.mlp", neurons = 10, batch_size = 16, epochs = 0), tsk("sonar"))
  expect_learner_torch(lrn("regr.mlp", neurons = 10, batch_size = 16, epochs = 0), tsk("mtcars"))
})
