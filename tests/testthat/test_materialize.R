test_that("materialize works on lazy_tensor", {
  ds = random_dataset(5, 4, n = 10)
  lt = as_lazy_tensor(ds, list(x = c(NA, 5, 4)))

  output = materialize(lt, device = "cpu", rbind = TRUE)
  expect_class(output, "torch_tensor")
  expect_equal(output$shape, c(10, 5, 4))
  expect_true(output$device == torch_device("cpu"))
  expect_true(torch_equal(output, ds$x))

  output_meta_list = materialize(lt, device = "meta", rbind = FALSE)
  output_meta_tnsr = materialize(lt, device = "meta", rbind = TRUE)

  expect_equal(torch_cat(output_meta_list, dim = 1L)$shape, output_meta_tnsr$shape)
  expect_true(output_meta_tnsr$device == torch_device("meta"))
})

test_that("materialize works in all 4 cases", {
  # .g
})

test_that("materialize works for data.frame", {
  df = nano_mnist()$data(1:10, cols = "image")

  out = materialize(df, rbind = TRUE)
  expect_list(out)
  expect_equal(names(out), "image")
  expect_class(out$image, "torch_tensor")
  expect_equal(out$image$shape, c(10, 1, 28, 28))
})

test_that("materialize works with differing shapes", {
  ds = dataset(
    initialize = function() {
      self$x = list(
        torch_randn(2, 2),
        torch_randn(3, 3)
      )
    },
    .getitem = function(i) {
      list(x = self$x[[i]])
    },
    .length = function() {
      length(self$x)
    }
  )()

  dd = DataDescriptor(ds, dataset_shapes = list(x = c(NA, NA, NA)))

  lt = as_lazy_tensor(dd)

  expect_class(materialize(lt[1], rbind = TRUE), "torch_tensor")
  l = materialize(lt, rbind = FALSE)
  expect_class(l, "list")
  expect_class(l[[1L]], "torch_tensor")
})

test_that("caching of datasets works", {
  # TODO
})

test_that("caching of graphs works", {
  # TODO
})


