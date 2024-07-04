test_that("PipeOpTorchIngressNumeric", {
  po_ingress = po("torch_ingress_num")
  dat = data.table(y = runif(10), x_cat = factor(letters[1:10]), x_lgl = TRUE, x_ord = ordered(letters[1:10]),
    x_num = runif(10), x_int = 1:10
  )
  task = as_task_regr(dat, target = "y")
  expect_po_ingress(po_ingress, task)
})

test_that("ingress fails with 0 features", {
  expect_error(
    po("torch_ingress_cat")$train(list(tsk("iris")))
  )

})

test_that("PipeOpTorchIngressCategorical", {
  po_ingress = po("torch_ingress_categ")
  dat = data.table(y = runif(10), x_cat = factor(letters[1:10]), x_lgl = TRUE, x_ord = ordered(letters[1:10]),
    x_num = runif(10), x_int = 1:10
  )
  task = as_task_regr(dat, target = "y")
  expect_po_ingress(po_ingress, task)
})

test_that("PipeOpTorchIngressImage", {
  po_ingress = po("torch_ingress_ltnsr")
  task = nano_imagenet()$cbind(data.frame(x1 = 1:10))
  expect_po_ingress(po_ingress, task)
})

test_that("PipeOpTorchIngressLazyTensor", {
  task = nano_mnist()
  po_ingress = po("torch_ingress_ltnsr")

  output = po_ingress$train(list(task))[[1L]]
  ds = task_dataset(task, output$ingress, device = "cpu",
    target_batchgetter = target_batchgetter_classif)

  batch = ds$.getbatch(1:2)
  expect_permutation(names(batch), c("x", "y", ".index"))
  expect_equal(names(batch$x), "torch_ingress_ltnsr.input")
  expect_class(batch$x[[1L]], "torch_tensor")
  expect_true(batch$x$torch_ingress_ltnsr.input$dtype == torch_float())
  expect_true(batch$x$torch_ingress_ltnsr.input$device == torch_device("cpu"))
  expect_equal(batch$x$torch_ingress_ltnsr.input$shape, c(2, 1, 28, 28))

  ds_meta = task_dataset(task, output$ingress, device = "meta")
  batch_meta = ds_meta$.getbatch(2:3)
  expect_true(batch_meta$x$torch_ingress_ltnsr.input$device == torch_device("meta"))

  task_old = task$clone()
  task$cbind(data.frame(row_id = 1:10, x_num = 1:10))
  expect_po_ingress(po_ingress, task)

  expect_error(po_ingress$param_set$set_values(shape = c(22, 4)))
  expect_error(po_ingress$param_set$set_values(shape = c(NA, 22, 4)), regexp = NA)
})

test_that("target can contain missings for ingress", {
  task = as_task_regr(data.table(y = c(1, NA), x = 1:1), target = "y")
  md = po("torch_ingress_num")$train(list(task))[[1L]]
  expect_class(md, "ModelDescriptor")
})

test_that("ingress cat: device placement", {
  task = tsk("german_credit")$select(c("job", "credit_history"))
  md = po("torch_ingress_categ")$train(list(task))[[1L]]
  iter = dataloader_make_iter(dataloader(task_dataset(task, md$ingress, device = "meta",
    target_batchgetter = target_batchgetter_regr)))
  batch = iter$.next()
  expect_equal(batch$x[[1L]]$device, torch_device("meta"))
  expect_equal(batch$y$device, torch_device("meta"))
  expect_equal(batch$.index$device, torch_device("meta"))
})

test_that("ingress num: device placement", {
  task = tsk("mtcars")
  md = po("torch_ingress_num")$train(list(task))[[1L]]
  iter = dataloader_make_iter(dataloader(task_dataset(task, md$ingress, device = "meta",
    target_batchgetter = target_batchgetter_classif)))
  batch = iter$.next()
  expect_equal(batch$x[[1L]]$device, torch_device("meta"))
  expect_equal(batch$y$device, torch_device("meta"))
  expect_equal(batch$.index$device, torch_device("meta"))
})

test_that("ingress ltnsr: device placement", {
  task = tsk("lazy_iris")
  md = po("torch_ingress_ltnsr")$train(list(task))[[1L]]
  iter = dataloader_make_iter(dataloader(task_dataset(task, md$ingress, device = "meta",
    target_batchgetter = target_batchgetter_regr)))
  batch = iter$.next()
  expect_equal(batch$x[[1L]]$device, torch_device("meta"))
  expect_equal(batch$y$device, torch_device("meta"))
  expect_equal(batch$.index$device, torch_device("meta"))
})
