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
    shape_exp = po_cat$shapes_out(list("..." = tensor1$shape, "..." = tensor2$shape))[[1L]]
    expect_true(all(out_obs$shape == shape_exp))

  }
})
