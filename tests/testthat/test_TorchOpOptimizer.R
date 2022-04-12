test_that("TorchOpOptimizer works", {
  opt = top("optimizer", .optimizer = "adam", lr = 0.1)
  opt$train(list(structure(class = "ModelArgs", list())))
})
