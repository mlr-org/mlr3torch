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
  dataset = random_dataset(10, 5, 3)
  expect_error(DataDescriptor$new(dataset, list(y = c(NA, 5, 3))), "must return a list")
  expect_error(DataDescriptor$new(dataset, list(x = c(NA, 10, 5, 4))), "First batch")
  expect_error(DataDescriptor$new(dataset, list(x = c(NA, 10, 5, 3)), input_map = "aaa"), "aaa")
  dataset1 = dataset(
    initialize = function() NULL,
    .getitem = function(i) list(x = torch_tensor(1), y = torch_tensor(1)),
    .length = function() 1)()
  expect_error(DataDescriptor$new(dataset1, list(x = c(NA, 10, 5, 3)), input_map = c("x", "y")),
    regexp = "that are a permutation")
  # dataset shapes must be provided
  expect_error(DataDescriptor$new(dataset), "missing")
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

