test_that("LearnerClassifTorch works with nn_module as architecture", {
  task = tsk("iris")
  net = nn_sequential$new(
    nn_tokenizer(4, cardinalities = integer(), d_token = 3, bias = TRUE, cls = FALSE),
    nn_flatten(),
    nn_linear(12, 10),
    nn_relu(),
    nn_linear(10, 4)
  )

  l = lrn("classif.torch", optimizer = "adam", criterion = "cross_entropy", architecture = net,
    optimizer_args = list(lr = 0.01), device = "cpu", epochs = 1L, batch_size = 16L
  )
  l$train(task)
  l$predict(task)

})

test_that("LearnerClassifTorch works with Architecture as architecture", {
  task = tsk("mtcars")
  graph = top("input") %>>%
    top("tokenizer", d_token = 5L) %>>%
    top("flatten") %>>%
    top("linear1", out_features = 10L) %>>%
    top("relu") %>>%
    top("linear2", out_features = 1L) %>>%
    top("")

  glrn = as_learner(graph)


  model_args = graph$train(task)[[1L]]
  network = model_args$architecture$build(model_args$task)

})
