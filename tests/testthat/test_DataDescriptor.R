test_that("DataDescriptor works", {
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

  dd = DataDescriptor$new(ds, dataset_shapes = list(x = c(NA, 5, 3)))
  expect_class(dd, "DataDescriptor")
  expect_equal(dd$pointer_shape, c(NA, 5, 3))
  expect_class(dd$graph$pipeops[[1L]], "PipeOpNOP")
  expect_true(length(dd$graph$pipeops) == 1L)
  expect_equal(dd$pointer, c(dd$graph$output$op.id, dd$graph$output$channel.name))
  expect_string(dd$dataset_hash)
  expect_string(dd$hash)
  expect_false(dd$dataset_hash == dd$hash)

  dd1 = DataDescriptor$new(ds, dataset_shapes = list(x = c(NA, 5, 3)))
  expect_equal(dd$dataset_shapes, dd1$dataset_shapes)

  # dataset shapes must be provided
  expect_error(DataDescriptor$new(ds), "missing")
  # batch must always be NA
  expect_error(DataDescriptor$new(ds, dataset_shapes = c(10, 5, 3)))

  graph = as_graph(po("nop", id = "nop"))

  expect_error(DataDescriptor$new(ds, dataset_shapes = list(x = c(NA, 5, 4)), "When passing a graph"))
})

