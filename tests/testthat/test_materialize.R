test_that("materialize works on lazy_tensor", {
  ds = random_dataset(5, 4, n = 10)
  lt = as_lazy_tensor(ds, list(x = c(NA, 5, 4)))

  output = materialize(lt, device = "cpu")
  expect_class(output, "torch_tensor")
  expect_equal(output$shape, c(10, 5, 4))
  expect_true(output$device == torch_device("cpu"))
  expect_true(torch_equal(output, ds$x))

  output_meta = materialize(lt, device = "meta")
  expect_true(output_meta$device == torch_device("meta"))
})

test_that("materialize works for data.frame", {
  df = nano_mnist()$data(1:10, cols = "image")

  out = materialize(df)
  expect_list(out)
  expect_equal(names(out), "image")
  expect_class(out$image, "torch_tensor")
  expect_equal(out$image$shape, c(10, 1, 28, 28))
})

test_that("caching of datasets works", {
  # TODO
})

test_that("caching of graphs works", {
  # TODO
})


