test_that("Basic checks", {
  ds = random_dataset(5, 3)
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
})

test_that("input verification", {
  dataset = make_dataset(list(x = c(10, 5, 3)), getbatch = FALSE)
  expect_error(DataDescriptor$new(dataset, list(y = c(NA, 5, 3))), "must return a list")
  expect_error(DataDescriptor$new(dataset, list(x = c(NA, 10, 5, 4))), "First batch")
  expect_error(DataDescriptor$new(dataset, list(x = c(NA, 10, 5, 3)), input_map = "aaa"), "aaa")
  dataset1 = make_dataset(list(x = 1, y = 1), getbatch = FALSE)
  expect_error(DataDescriptor$new(dataset1, list(x = c(NA, 10, 5, 3)), input_map = c("x", "y")),
    regexp = "that are a permutation")
  # dataset shapes must be provided
  expect_error(DataDescriptor$new(dataset))
  # batch must always be NA
  expect_error(DataDescriptor$new(dataset, dataset_shapes = c(10, 10, 5, 3)))

  graph = as_graph(po("nop", id = "nop"))
  expect_error(DataDescriptor$new(ds, dataset_shapes = list(x = c(NA, 5, 4)), "When passing a graph"))
})

test_that("can infer the simple case", {
  dataset = random_dataset(5)
  dd = DataDescriptor$new(dataset, list(x = c(NA, 5)))
  expect_equal(dd$input_map, "x")
  expect_equal(dd$pointer_shape, c(NA, 5))
  expect_equal(dd$pointer_shape_predict, c(NA, 5))
  expect_equal(dd$pointer, c(dd$graph$output$op.id, dd$graph$output$channel.name))
})


test_that("printer", {
  dd = DataDescriptor$new(random_dataset(1), list(x = NULL))
  # unknown shapes
  expect_true(grepl(paste0(capture.output(dd), collapse = ""), pattern = "<unknown>", fixed = TRUE))
})

test_that("infer shapes from dataset", {
  dd = DataDescriptor$new(random_dataset(3, 2))
  expect_equal(dd$dataset_shapes, list(x = c(NA, 3L, 2L)))

  ds1 = dataset("test",
    initialize = function() {
      self$x = 1
    },
    .getitem = function(i) {
      list(x = torch_tensor(self$x))
    },
    .length = function(i) 1L
  )()
  expect_error(DataDescriptor$new(ds1), "must be provided if dataset does not")

  ds = make_dataset(list(1))

  expect_error(DataDescriptor$new(ds), "must return a named list of tensors")
})

test_that("assert_compatible_shapes", {
  ds = make_dataset(list(x = c(2, 3)), getbatch = TRUE)
  expect_error(assert_compatible_shapes(list(x = c(NA, 2, 3)), ds), regexp = NA)
  expect_error(assert_compatible_shapes(list(x = c(NA, 2, 1)), ds), regexp = "(NA,2,1)")
  ds = make_dataset(list(x = c(2, 3)), getbatch = FALSE)
  expect_error(assert_compatible_shapes(list(x = c(NA, 2, 3)), ds), regexp = NA)
  expect_error(assert_compatible_shapes(list(x = c(NA, 2, 1)), ds), regexp = "(NA,2,1)")
  ds = make_dataset(list(x = c(2, 3), y = 1), getbatch = TRUE)
  expect_error(assert_compatible_shapes(list(x = c(NA, 2, 3), y = c(NA, 1)), ds), regexp = NA)
  expect_error(assert_compatible_shapes(list(x = c(NA, 2, 3), y = c(NA, 2)), ds), regexp = "shape of y")
  ds = make_dataset(list(x = c(2, 3), y = 1), getbatch = FALSE)
  expect_error(assert_compatible_shapes(list(x = c(NA, 2, 3), y = c(NA, 1)), ds), regexp = NA)
  expect_error(assert_compatible_shapes(list(x = c(NA, 2, 3), y = c(NA, 2)), ds), regexp = "shape of y")
})

test_that("as_data_descriptor", {
  ds = make_dataset(list(x = 1))
  expect_class(as_data_descriptor(ds), "DataDescriptor")
})
