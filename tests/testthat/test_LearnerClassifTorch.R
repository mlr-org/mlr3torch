test_that("LearnerClassifTorch works with nn_module as architecture", {
  task = tsk("iris")
  net = nn_sequential$new(
    nn_tokenizer(4, cardinalities = integer(), d_token = 3, bias = TRUE, cls = FALSE),
    nn_flatten(),
    nn_linear(12, 10),
    nn_relu(),
    nn_linear(10, 4)
  )

  l = lrn("classif.torch", .optimizer = "adam", criterion = "cross_entropy", architecture = net,
    lr = 0.1, device = "cpu", epochs = 2L, batch_size = 16L
  )
  l$train(task)
  expect_error(l$train(task), regexp = NA)
  expect_error(l$predict(task), regexp = NA)
})

test_that("LearnerClassifTorch works with Architecture as architecture", {
  task = tsk("iris")
  graph = top("input") %>>%
    top("tokenizer", d_token = 5L) %>>%
    top("flatten") %>>%
    top("linear1", out_features = 10L) %>>%
    top("relu") %>>%
    top("linear2", out_features = 1L) %>>%
    top("model.classif", .optimizer = "adam", lr = 0.01, criterion = "cross_entropy",
      batch_size = 1L
    )
  glrn = as_learner(graph)
  glrn$train(task)

  model_args = graph$train(task)[[1L]]
  network = model_args$architecture$build(model_args$task)

})
