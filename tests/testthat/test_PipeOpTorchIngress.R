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
  ds = task_dataset(task, output$ingress, target_batchgetter = target_batchgetter_classif_binary)

  batch = ds$.getbatch(1:2)
  expect_permutation(names(batch), c("x", "y", ".index"))
  expect_equal(names(batch$x), "torch_ingress_ltnsr.input")
  expect_class(batch$x[[1L]], "torch_tensor")
  expect_true(batch$x$torch_ingress_ltnsr.input$dtype == torch_float())
  expect_equal(batch$x$torch_ingress_ltnsr.input$shape, c(2, 1, 28, 28))

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


test_that("shape of lazy tensor ingress can be inferred", {
  po_ingress = po("torch_ingress_ltnsr", shape = "infer")
  out = po_ingress$train(list(nano_dogs_vs_cats()))[[1L]]
  expect_equal(out$pointer_shape, c(NA, 3, 280, 300))
})

test_that("error message unknown shapes", {
  task = po("augment_random_crop", size = c(100, 100))$train(list(nano_imagenet()))[[1L]]
  obj = po("torch_ingress_ltnsr")
  expect_error(obj$train(list(task)), "see its documentation")
})

test_that("allow for flexible shapes", {
  ingress = po("torch_ingress_ltnsr", shape = c(NA, NA))
  task
})
