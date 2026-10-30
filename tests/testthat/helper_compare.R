# testthat's expect_equal() uses waldo::compare(), which compares torch objects (external pointers)
# by address. torch ships a compare_proxy() method for tensors, but it compares as_array(x), which
# ignores the dtype and (partly) the shape, and it compares grad_fn by address, so two tensors that
# require grad are never equal. We replace it: tensors are equal if dtype, shape, device,
# requires_grad and values agree. `tolerance` applies to the values.
compare_proxy_torch_tensor = function(x, path) {
  device = as.character(x$device)
  values = if (device == "meta") NULL else as.vector(torch::as_array(x$detach()$cpu()))
  list(
    object = list(
      dtype = as.character(x$dtype), shape = x$shape, device = device,
      requires_grad = x$requires_grad, values = values
    ),
    path = path
  )
}

compare_proxy_as_character = function(x, path) {
  list(object = as.character(x), path = sprintf("as.character(%s)", path))
}

registerS3method("compare_proxy", "torch_tensor", compare_proxy_torch_tensor, envir = asNamespace("waldo"))
registerS3method("compare_proxy", "torch_dtype", compare_proxy_as_character, envir = asNamespace("waldo"))
registerS3method("compare_proxy", "torch_device", compare_proxy_as_character, envir = asNamespace("waldo"))
