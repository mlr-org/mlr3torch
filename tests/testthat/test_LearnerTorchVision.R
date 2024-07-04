# different number of classes than the predefined ones
task = as_task_classif(data.table(
  y = as.factor(rep(c("a", "b", "c"), each = 2)),
  x = as_lazy_tensor(torch_randn(6, 3, 64, 64))
), id = "test_task", target = "y")

test_that("LearnerTorchVision basic checks", {
  alexnet = lrn("classif.alexnet", epochs = 1L, batch_size = 1L, pretrained = FALSE)
  expect_deep_clone(alexnet, alexnet$clone(deep = TRUE))

  alexnet$train(task)
  expect_class(alexnet$predict(task), "PredictionClassif")

  expect_learner_torch(alexnet, task = task)
  alexnet$id = "a"
  vgg13 = lrn("classif.vgg13", pretrained = FALSE)
  vgg13$id = "a"
  expect_false(alexnet$phash == vgg13$phash)
  expect_true("torchvision" %in% alexnet$packages)
  expect_true("magick" %in% alexnet$packages)

  alexnet = lrn("classif.alexnet", optimizer = "sgd", loss = "cross_entropy",
    callbacks = t_clbk("checkpoint"), epochs = 0, batch_size = 1
  )
  expect_learner(alexnet)
  expect_true("cb.checkpoint.freq" %in% alexnet$param_set$ids())
})

test_that("alexnet", {
  learner = lrn("classif.alexnet", epochs = 0L, batch_size = 2L, pretrained = FALSE)
  learner$train(task, sample(task$nrow, 1L))
  pred = learner$predict(task, sample(task$nrow, 1L))
  expect_class(pred, "PredictionClassif")
})

# weird warning regarding weight initialization from torchvision
# test_that("inception_v3", {
#   learner = lrn("classif.inception_v3", epochs = 0L, batch_size = 2L, pretrained = FALSE)
#   learner$train(task, sample(task$nrow, 1L))
#   pred = learner$predict(task, sample(task$nrow, 1L))
#   expect_class(pred, "PredictionClassif")
# })

# these tests are run the CI, but they should basically never fail, so
# we skip them in the local run
# models are also cached in the CI, so it is not too slow
skip_if(!identical(Sys.getenv("INCLUDE_IGNORED"),  "1"), "Slow vision tests")

test_that("mobilenet_v2", {
  learner = lrn("classif.mobilenet_v2", epochs = 0L, batch_size = 2L, pretrained = FALSE)
  learner$train(task, sample(task$nrow, 1L))
  pred = learner$predict(task, sample(task$nrow, 1L))
  expect_class(pred, "PredictionClassif")
  learner = lrn("classif.mobilenet_v2", epochs = 0L, batch_size = 2L, pretrained = TRUE)
  learner$train(task, sample(task$nrow, 1L))
  pred = learner$predict(task, sample(task$nrow, 1L))
  expect_class(pred, "PredictionClassif")
})

test_that("resnet18", {
  learner = lrn("classif.resnet18", epochs = 0L, batch_size = 2L, pretrained = FALSE)
  learner$train(task, sample(task$nrow, 1L))
  pred = learner$predict(task, sample(task$nrow, 1L))
  expect_class(pred, "PredictionClassif")

  learner = lrn("classif.resnet18", epochs = 0L, batch_size = 2L, pretrained = TRUE)
  learner$train(task, sample(task$nrow, 1L))
  pred = learner$predict(task, sample(task$nrow, 1L))
  expect_class(pred, "PredictionClassif")
})

test_that("resnet34", {
  learner = lrn("classif.resnet34", epochs = 0L, batch_size = 2L, pretrained = FALSE)
  learner$train(task, sample(task$nrow, 1L))
  pred = learner$predict(task, sample(task$nrow, 1L))
  expect_class(pred, "PredictionClassif")

  learner = lrn("classif.resnet34", epochs = 0L, batch_size = 2L, pretrained = TRUE)
  learner$train(task, sample(task$nrow, 1L))
  pred = learner$predict(task, sample(task$nrow, 1L))
  expect_class(pred, "PredictionClassif")
})

test_that("resnet50", {
  learner = lrn("classif.resnet50", epochs = 0L, batch_size = 2L, pretrained = FALSE)
  learner$train(task, sample(task$nrow, 1L))
  pred = learner$predict(task, sample(task$nrow, 1L))
  expect_class(pred, "PredictionClassif")

  learner = lrn("classif.resnet50", epochs = 0L, batch_size = 2L, pretrained = TRUE)
  learner$train(task, sample(task$nrow, 1L))
  pred = learner$predict(task, sample(task$nrow, 1L))
  expect_class(pred, "PredictionClassif")
})

test_that("resnet101", {
  learner = lrn("classif.resnet101", epochs = 0L, batch_size = 2L, pretrained = FALSE)
  learner$train(task, sample(task$nrow, 1L))
  pred = learner$predict(task, sample(task$nrow, 1L))
  expect_class(pred, "PredictionClassif")

  learner = lrn("classif.resnet101", epochs = 0L, batch_size = 2L, pretrained = TRUE)
  learner$train(task, sample(task$nrow, 1L))
  pred = learner$predict(task, sample(task$nrow, 1L))
  expect_class(pred, "PredictionClassif")
})

test_that("resnet152", {
  learner = lrn("classif.resnet152", epochs = 0L, batch_size = 2L, pretrained = FALSE)
  learner$train(task, sample(task$nrow, 1L))
  pred = learner$predict(task, sample(task$nrow, 1L))
  expect_class(pred, "PredictionClassif")

  learner = lrn("classif.resnet152", epochs = 0L, batch_size = 2L, pretrained = TRUE)
  learner$train(task, sample(task$nrow, 1L))
  pred = learner$predict(task, sample(task$nrow, 1L))
  expect_class(pred, "PredictionClassif")
})

test_that("resnet101_32x8d", {
  learner = lrn("classif.resnext101_32x8d", epochs = 0L, batch_size = 2L, pretrained = FALSE)
  learner$train(task, sample(task$nrow, 1L))
  pred = learner$predict(task, sample(task$nrow, 1L))
  expect_class(pred, "PredictionClassif")

  learner = lrn("classif.resnext101_32x8d", epochs = 0L, batch_size = 2L, pretrained = TRUE)
  learner$train(task, sample(task$nrow, 1L))
  pred = learner$predict(task, sample(task$nrow, 1L))
  expect_class(pred, "PredictionClassif")
})

test_that("resnet50_32x4d", {
  learner = lrn("classif.resnext50_32x4d", epochs = 0L, batch_size = 2L, pretrained = FALSE)
  learner$train(task, sample(task$nrow, 1L))
  pred = learner$predict(task, sample(task$nrow, 1L))
  expect_class(pred, "PredictionClassif")

  learner = lrn("classif.resnext50_32x4d", epochs = 0L, batch_size = 2L, pretrained = TRUE)
  learner$train(task, sample(task$nrow, 1L))
  pred = learner$predict(task, sample(task$nrow, 1L))
  expect_class(pred, "PredictionClassif")
})

test_that("vgg11", {
  learner = lrn("classif.vgg11", epochs = 0L, batch_size = 2L, pretrained = FALSE)
  learner$train(task, sample(task$nrow, 1L))
  pred = learner$predict(task, sample(task$nrow, 1L))
  expect_class(pred, "PredictionClassif")

  learner = lrn("classif.vgg11", epochs = 0L, batch_size = 2L, pretrained = TRUE)
  learner$train(task, sample(task$nrow, 1L))
  pred = learner$predict(task, sample(task$nrow, 1L))
  expect_class(pred, "PredictionClassif")
})

test_that("vgg11_bn", {
  learner = lrn("classif.vgg11_bn", epochs = 0L, batch_size = 2L, pretrained = FALSE)
  learner$train(task, sample(task$nrow, 1L))
  pred = learner$predict(task, sample(task$nrow, 1L))
  expect_class(pred, "PredictionClassif")

  learner = lrn("classif.vgg11_bn", epochs = 0L, batch_size = 2L, pretrained = TRUE)
  learner$train(task, sample(task$nrow, 1L))
  pred = learner$predict(task, sample(task$nrow, 1L))
  expect_class(pred, "PredictionClassif")
})

test_that("vgg13", {
  learner = lrn("classif.vgg13", epochs = 0L, batch_size = 2L, pretrained = FALSE)
  learner$train(task, sample(task$nrow, 1L))
  pred = learner$predict(task, sample(task$nrow, 1L))
  expect_class(pred, "PredictionClassif")

  learner = lrn("classif.vgg13", epochs = 0L, batch_size = 2L, pretrained = TRUE)
  learner$train(task, sample(task$nrow, 1L))
  pred = learner$predict(task, sample(task$nrow, 1L))
  expect_class(pred, "PredictionClassif")
})

test_that("vgg13_bn", {
  learner = lrn("classif.vgg13_bn", epochs = 0L, batch_size = 2L, pretrained = FALSE)
  learner$train(task, sample(task$nrow, 1L))
  pred = learner$predict(task, sample(task$nrow, 1L))
  expect_class(pred, "PredictionClassif")

  learner = lrn("classif.vgg13_bn", epochs = 0L, batch_size = 2L, pretrained = TRUE)
  learner$train(task, sample(task$nrow, 1L))
  pred = learner$predict(task, sample(task$nrow, 1L))
  expect_class(pred, "PredictionClassif")
})

test_that("vgg16", {
  learner = lrn("classif.vgg16", epochs = 0L, batch_size = 2L, pretrained = FALSE)
  learner$train(task, sample(task$nrow, 1L))
  pred = learner$predict(task, sample(task$nrow, 1L))
  expect_class(pred, "PredictionClassif")

  learner = lrn("classif.vgg16", epochs = 0L, batch_size = 2L, pretrained = TRUE)
  learner$train(task, sample(task$nrow, 1L))
  pred = learner$predict(task, sample(task$nrow, 1L))
  expect_class(pred, "PredictionClassif")
})

test_that("vgg16_bn", {
  learner = lrn("classif.vgg16_bn", epochs = 0L, batch_size = 2L, pretrained = FALSE)
  learner$train(task, sample(task$nrow, 1L))
  pred = learner$predict(task, sample(task$nrow, 1L))
  expect_class(pred, "PredictionClassif")

  learner = lrn("classif.vgg16_bn", epochs = 0L, batch_size = 2L, pretrained = TRUE)
  learner$train(task, sample(task$nrow, 1L))
  pred = learner$predict(task, sample(task$nrow, 1L))
  expect_class(pred, "PredictionClassif")
})

test_that("vgg19", {
  learner = lrn("classif.vgg19", epochs = 0L, batch_size = 2L, pretrained = FALSE)
  learner$train(task, sample(task$nrow, 1L))
  pred = learner$predict(task, sample(task$nrow, 1L))
  expect_class(pred, "PredictionClassif")

  learner = lrn("classif.vgg19", epochs = 0L, batch_size = 2L, pretrained = TRUE)
  learner$train(task, sample(task$nrow, 1L))
  pred = learner$predict(task, sample(task$nrow, 1L))
  expect_class(pred, "PredictionClassif")
})

test_that("vgg19_bn", {
  learner = lrn("classif.vgg19_bn", epochs = 0L, batch_size = 2L, pretrained = FALSE)
  learner$train(task, sample(task$nrow, 1L))
  pred = learner$predict(task, sample(task$nrow, 1L))
  expect_class(pred, "PredictionClassif")

  learner = lrn("classif.vgg19_bn", epochs = 0L, batch_size = 2L, pretrained = TRUE)
  learner$train(task, sample(task$nrow, 1L))
  pred = learner$predict(task, sample(task$nrow, 1L))
  expect_class(pred, "PredictionClassif")
})
