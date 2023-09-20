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

test_that("PipeOpTorchIngressLazyTensor works", {
  ds = dataset(
    initialize = function() {
      self$x = torch_randn(10, 5, 3)
    },
    .getitem = function(i) {
      list(x = self$x[i, ..])
    },

    .length = function() {
      nrow(self$x)
    }
  )()

  dd = DataDescriptor(ds, shapes = list(x = c(5, 3)))

  ltnsr = lazy_tensor(dd)

  b = data.table(
    y = rnorm(10),
    x = ltnsr
  )

  task = as_task_regr(b, target = "y")

  obj = po("torch_ingress_ltnsr", to_tensor = FALSE)

  md = obj$train(list(task))[[1L]]
  expect_equal(md$.pointer_shape, c(NA, 5, 3))

  batch = md$ingress[[1L]]$batchgetter(task$data(1:2, "x"), "cpu")
  expect_true(length(batch) == 2)
  expect_class(batch, "lazy_tensor")

  graph = po("torch_ingress_ltnsr", to_tensor = FALSE) %>>%
    po("transform_resize", size = c(10, 10)) %>>%
    po("torch_tensor") %>>%
    po("nn_conv1d", kernel_size = 3L, out_channels = 2L)

  output = graph$train(task)[[1L]]
  ds = task_dataset(output$task, output$ingress, device = "cpu")
  network = output$graph
  debugonce(ds$.getbatch)
  batch1 = ds$.getbatch(1:2)
  # here we don't concatenate
  expect_equal(batch1$x[[1L]]$shape, c(5, 3))

  graph$param_set$set_values(to_tensor = TRUE)
})
