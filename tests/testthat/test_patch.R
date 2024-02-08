test_that("patch for cloning works", {
  lin = nn_linear(1, 1)
  lin2 = lin$clone(deep = TRUE)
  params = lin$parameters
  params2 = lapply(params, function(p) p$clone()$detach())

  params2 = patch_list_clone(old = params, new = params2) # in-place
  expect_equal(params[[1]]$requires_grad, params2[[1]]$requires_grad)

  expect_false(lin$parameters[[1]]$requires_grad == lin2$parameters[[1]]$requires_grad)
  patch_module_clone(lin, lin2)
  expect_equal(lin$parameters[[1]]$requires_grad, lin2$parameters[[1]]$requires_grad)
})

test_that("expect_equal for tensors", {
  expect_equal(torch_tensor(1), torch_tensor(1))
})
