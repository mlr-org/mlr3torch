compare_proxy.torch_tensor = function(x, path) {
  list(
    object = list(x = torch::as_array(x), device = as.character(x$device), requires_grad = x$requires_grad),
    path = path
  )
}

patch_module_clone = function(old, new) {
  patch_list_clone(old$parameters, new$parameters)
  new
}

patch_list_clone = function(old, new) {
  walk(names(old), function(nm) {
    new[[nm]]$detach_()$requires_grad_(old[[nm]]$requires_grad)
  })
  new
}
