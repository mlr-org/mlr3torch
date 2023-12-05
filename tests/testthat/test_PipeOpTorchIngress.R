test_that("PipeOpTorchIngressNumeric", {
  po_ingress = po("torch_ingress_num")
  dat = data.table(y = runif(10), x_cat = factor(letters[1:10]), x_lgl = TRUE, x_ord = ordered(letters[1:10]),
    x_num = runif(10), x_int = 1:10
  )
  task = as_task_regr(dat, target = "y")
  expect_po_ingress(po_ingress, task)
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
  po_ingress = po("torch_ingress_img", channels = 3, width = 64, height = 64)
  task = nano_imagenet()$cbind(data.frame(x1 = 1:10))
  expect_po_ingress(po_ingress, task)
})

test_that("PipeOpTorchIngressLazyTensor", {
  task = nano_mnist()
  po_ingress = po("torch_ingress_ltnsr")

  output = po_ingress$train(list(task))[[1L]]
  ds = task_dataset(task, output$ingress, device = "cpu")

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
})
