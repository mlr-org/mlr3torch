test_that("lazy_tensor works", {
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

  tds = torch_dataset(shapes = list(x = 5), dataset = ds)
  dd = DataDescriptor(dsd)

  ltnsr = lazy_tensor(dd)

  expect_equal(length(ltnsr), 10)
})

test_that("PreprocDescriptor works", {
  ds = dataset(
    initialize = function() {
      self$x = torch_randn(10, 5)
    },
    .getitem = function(i) {
      list(x = self$x[i, ..])
    },
    .length = function() {
      nrow(self$x)
    }
  )()

  dsd = torch_dataset(shapes = list(x = 5), dataset = ds)

  ltnsr = lazy_tensor(dsd)


})
