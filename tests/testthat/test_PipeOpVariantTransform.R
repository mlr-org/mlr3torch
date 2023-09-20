test_that("PipeOpVariantTransform works", {
  ds = dataset(
    "mydata",
    initialize = function() {
      self$x = torch_ones(10, 5, 3)
    },
    .getitem = function(i) {
      list(x = self$x[i, ..])
    },
    .length = function() {
      nrow(self$x)
    }
  )()

  task = nano_dogs_vs_cats()

  trafo1 = po("transform_resize", size = c(10, 4))


  # FIXME: better printer for tasks (shape of lazy tensor)
  task1 = trafo1$train(list(task))[[1L]]


})
