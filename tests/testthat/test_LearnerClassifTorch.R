test_that("LearnerClassifTorch works with nn_module as architecture", {
  task = tsk("iris")
  net = nn_sequential$new(
    nn_tokenizer(4, cardinalities = integer(), d_token = 3, bias = TRUE, cls = FALSE),
    nn_flatten(),
    nn_linear(12, 10),
    nn_relu(),
    nn_linear(10, 4)
  )

  l = lrn("classif.torch", .optimizer = "adam", .loss = "cross_entropy", .network = net,
    opt.lr = 0.1, device = "cpu", epochs = 2L, batch_size = 16L
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
    top("linear_1", out_features = 10L) %>>%
    top("relu") %>>%
    top("linear_2", out_features = 4L)

  network = graph$train(task)[[1L]]$network
  l = lrn("classif.torch", .optimizer = "adam", .loss = "cross_entropy",
    .network = network, opt.lr = 0.1, device = "cpu", epochs = 1L, batch_size = 1L
  )
  expect_error(l$train(task), regexp = NA)
  expect_error(l$predict(task), regexp = NA)

})
