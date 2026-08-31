test_that("PipeOpTorchMergeSum autotest", {
  po_test = po("nn_merge_sum")
  task = tsk("iris")
  graph = po("torch_ingress_num") %>>%
    list(po("nn_linear_1", out_features = 10), po("nn_linear_2", out_features = 10)) %>>%
    po_test

  expect_pipeop_torch(graph, "nn_merge_sum", task)
})

test_that("basic test", {
  # FIXME: this failed earlier because of a PipeOpTorch bug probably remove this test later
  task = tsk("iris")
  graph = pos(c("torch_ingress_num_1", "torch_ingress_num_2")) %>>% po("nn_merge_sum", innum = 2)

  md = graph$train(task)[[1L]]
  expect_class(md, "ModelDescriptor")
})

test_that("PipeOpTorchMergeSum paramtest", {
  po_test = po("nn_merge_sum")
  res = expect_paramset(po_test, nn_merge_sum)
  expect_paramtest(res)
})

test_that("PipeOpTorchMergeProd autotest", {
  po_test = po("nn_merge_prod")
  task = tsk("iris")
  graph = po("torch_ingress_num") %>>%
    list(po("nn_linear_1", out_features = 10), po("nn_linear_2", out_features = 10)) %>>%
    po_test

  expect_pipeop_torch(graph, "nn_merge_prod", task)
})

test_that("PipeOpTorchMergeProd paramtest", {
  po_test = po("nn_merge_prod")
  res = expect_paramset(po_test, nn_merge_prod)
  expect_paramtest(res)
})


test_that("PipeOpTorchMergeCat autotest", {
  po_test = po("nn_merge_cat", dim = 2)
  task = tsk("iris")
  graph = po("torch_ingress_num") %>>%
    list(po("nn_linear_1", out_features = 10), po("nn_linear_2", out_features = 10)) %>>%
    po_test

  expect_pipeop_torch(graph, "nn_merge_cat", task)

})


test_that("PipeOpTorchMergeCat paramtest", {
  po_test = po("nn_merge_cat")
  res = expect_paramset(po_test, nn_merge_cat)
  expect_paramtest(res)
})

sampler_merge = function() {
  d_max = sample(5, 1)
  n_dim = sample(1:5, 1)
  c(NA, sample(seq(d_max), replace = TRUE, size = n_dim))
}


test_that("Broadcasting is implemented correctly for prod and sum", {
  po_sum = po("nn_merge_sum")
  po_prod = po("nn_merge_sum")

  net_sum = nn_merge_sum()
  net_prod = nn_merge_prod()

  expect_error(po_sum$shapes_out(list(c(1, 2, 3), c(2, 3))))
  expect_error(po_prod$shapes_out(list(c(1, 2, 3), c(2, 3))))
  for (i in 1:10) {
    batch_size = sample(5, size = 1)

    shape1 = sampler_merge()
    shape2 = shape1

    # we set some of the dimensions to 1 (not the batch dimension though) to check that broadcasting is correctly
    # applied.
    ii = c(FALSE, sample(c(TRUE, FALSE), replace = TRUE, size = length(shape2) - 1))
    shape2[ii] = 1

    # the copy has an actual batch size so we can generate tensors with which to verify our implementation
    shape1copy = shape1
    shape1copy[1] = batch_size
    shape2copy = shape2
    shape2copy[1] = batch_size

    tensor1 = invoke(torch_randn, .args = as.list(shape1copy), device = torch_device("meta"))
    tensor2 = invoke(torch_randn, .args = as.list(shape2copy), device = torch_device("meta"))

    out1 = net_sum(tensor1, tensor2)
    out2 = net_prod(tensor1, tensor2)

    # now we check that the shapes agree
    observed1 = po_sum$shapes_out(list(input1 = shape1, input2 = shape2))[[1L]]
    observed1[1] = batch_size

    observed2 = po_prod$shapes_out(list(input1 = shape1, input2 = shape2))[[1L]]
    observed2[1] = batch_size

    expect_true(all(out1$shape == observed1))
    expect_true(all(out2$shape == observed2))

    # Here we check that an error is thrown if there is a dimension (i.e. the second dimension) that does not match
    shape1[2] = 100
    shape2[2] = 101
    expect_error(po_test$shapes_out(list(input1 = shape1, input2 = shape2)))
  }
})

test_that("Broadcasting is correctly implemented for concatenation", {
  po_cat = po("nn_merge_cat", dim = 2)
  net_cat = nn_merge_cat(dim = 2)

  for (i in 1:10) {

    batch_size = sample(5, size = 1)
    shape = sampler_merge()
    shape1 = c(shape[1], 7, tail(shape, -1))
    shape2 = c(shape[1], 8, tail(shape, -1))

    # The cat operator does not do broadcasting!
    shape1[1] = batch_size
    shape2[1] = batch_size

    tensor1 = invoke(torch_randn, .args = shape1, device = torch_device("meta"))
    tensor2 = invoke(torch_randn, .args = shape2, device = torch_device("meta"))

    out_obs = net_cat(tensor1, tensor2)
    shape_exp = po_cat$shapes_out(set_names(list(tensor1$shape, tensor2$shape), c("...", "...")))[[1L]]
    expect_true(all(out_obs$shape == shape_exp))

  }
})

test_that("merge infers the broadcast shape, not the shape of the first input", {
  # the output is the broadcast of the inputs, so a (NA,1) + (NA,6) merge produces (NA,6), which
  # is what the next layer is built for
  expect_equal(po("nn_merge_sum", innum = 2)$shapes_out(list(c(NA, 1L), c(NA, 6L)))[[1L]],
    c(NA, 6L))
  expect_equal(po("nn_merge_prod", innum = 2)$shapes_out(list(c(NA, 6L), c(NA, 1L)))[[1L]],
    c(NA, 6L))

  # a known size is not lost when another input is unknown in that dimension
  expect_equal(po("nn_merge_sum", innum = 2)$shapes_out(list(c(NA, NA, 4L), c(NA, 3L, 4L)))[[1L]],
    c(NA, 3L, 4L))
  # ... but stays unknown when every input is unknown there
  expect_equal(po("nn_merge_sum", innum = 2)$shapes_out(list(c(NA, NA, 4L), c(NA, NA, 4L)))[[1L]],
    c(NA, NA, 4L))

  expect_error(po("nn_merge_sum", innum = 2)$shapes_out(list(c(NA, 3L), c(NA, 5L))),
    "incompatible sizes")
  expect_error(po("nn_merge_sum", innum = 2)$shapes_out(list(c(NA, 3L), c(NA, 3L, 4L))),
    "same number of dimensions")
})

test_that("shape inference matches the operator", {
  expect_shape_inference("nn_merge_sum", list(), c(2, 4, 6), n_in = 2L)
  expect_shape_inference("nn_merge_prod", list(), c(2, 4, 6), n_in = 2L)
  expect_shape_inference("nn_merge_cat", list(dim = 2), c(2, 4, 6), n_in = 2L)
})

test_that("nn_merge_cat requires the other dimensions to be equal", {
  cat_shapes_out = function(...) po("nn_merge_cat", dim = 2)$shapes_out(list(...))[[1L]]
  expect_equal(cat_shapes_out(c(NA, 4L, 6L), c(NA, 5L, 6L)), c(NA, 9L, 6L))
  # an unknown dimension is determined by the known one, because the two must be equal
  expect_equal(cat_shapes_out(c(NA, 4L, NA), c(NA, 4L, 6L)), c(NA, 8L, 6L))
  # ... and an unknown size along the concatenated dimension makes the sum unknown
  expect_equal(cat_shapes_out(c(NA, NA, 6L), c(NA, 4L, 6L)), c(NA, NA, 6L))

  # `torch_cat()` does not broadcast, so a size of 1 does not combine with a different size
  expect_error(cat_shapes_out(c(NA, 4L, 1L), c(NA, 4L, 6L)),
    "dimension 3 has the sizes 1 and 6", fixed = TRUE)
  # the error names the shapes the PipeOp was given
  expect_error(cat_shapes_out(c(NA, 4L, 5L), c(NA, 4L, 6L)), "[(NA,4,5);(NA,4,6)]", fixed = TRUE)
})

test_that("shape inference agrees with the module for random shapes and parameters", {
  expect_shape_inference("nn_merge_sum", generators = gen_shape(3L), n_in = 2L)
  expect_shape_inference("nn_merge_prod", generators = gen_shape(3L), n_in = 2L)
  expect_shape_inference("nn_merge_cat", params = function() list(dim = sample(2:3, 1L)),
    generators = gen_shape(3L), n_in = 2L)
})
