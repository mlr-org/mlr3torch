test_that("LearnerClassifMLP works", {
  # TODO: Test make_mlp() separately

  verify_network = function(learner) {
    tab = table(map_chr(learner$network$children, function(x) class(x)[[1L]]))
    act = learner$param_set$values$activation

    l = learner$param_set$values$layers

    expect_true(tab["nn_linear"] == l + 1)
    if (l > 0) {
      expect_true(tab[paste0("nn_", learner$param_set$values$activation)] == l)
      expect_true(tab["nn_dropout"] == l)
    } else {
      expect_true(length(tab) == 1)
    }
  }

  learner = lrn("classif.mlp",
    layers = 2L,
    p = 0.111,
    batch_size = 16L,
    epochs = 0,
    d_hidden = 13,
    optimizer = "adagrad",
    activation = "softshrink",
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

  learner$param_set$set_values(layers = 0L)

  learner$train(task)
  verify_network(learner)
})

test_that("autotest", {
  learner = lrn("classif.mlp",
    layers = 2L,
    p = 0.2,
    batch_size = 16L,
    epochs = 10,
    d_hidden = 10,
    activation = "relu",
    optimizer = "adam",
    seed = 1
  )

  result = run_autotest(learner, check_replicable = TRUE, exclude = "sanity")

  expect_true(result, info = result$error)
})


test_that("LearnerRegrMLP works", {

  verify_network = function(learner) {
    tab = table(map_chr(learner$network$children, function(x) class(x)[[1L]]))
    act = learner$param_set$values$activation

    l = learner$param_set$values$layers

    expect_true(tab["nn_linear"] == l + 1)
    if (l > 0) {
      expect_true(tab[paste0("nn_", learner$param_set$values$activation)] == l)
      expect_true(tab["nn_dropout"] == l)
    } else {
      expect_true(length(tab) == 1)
    }
  }

  learner = lrn("regr.mlp",
    layers = 2L,
    p = 0.111,
    batch_size = 16L,
    epochs = 0,
    d_hidden = 13,
    optimizer = "adagrad",
    activation = "softshrink",
    activation_args = list(lambd = 0.25)
  )
  task = tsk("mtcars")
  learner$train(task)

  expect_true(nrow(learner$network$children[[1L]]$weight) == 13L)
  expect_true(!is.null(learner$network$children[[1L]]$bias))
  expect_true(learner$network$children[[2L]]$lambd == 0.25)

  expect_learner(learner)
  expect_class(learner$network, c("nn_sequential", "nn_module"))

  verify_network(learner)

  learner$param_set$set_values(layers = 0L)

  learner$train(task)
  verify_network(learner)
})

test_that("autotest", {
  learner = lrn("regr.mlp",
    layers = 2L,
    p = 0.2,
    batch_size = 20L,
    epochs = 10,
    d_hidden = 10,
    activation = "relu",
    optimizer = t_opt("sgd", lr = 0.01),
    seed = 1
  )

  result = run_autotest(learner, check_replicable = TRUE, exclude = "sanity")

  expect_true(result, info = result$error)
})
