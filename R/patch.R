compare_proxy.torch_tensor = function(x, path) {
  # we don't include the grad_fn because it should change after cloning a tensor (BackwardClone)
  list(
    object = list(
      x = torch::as_array(x),
      device = as.character(x$device),
      requires_grad = x$requires_grad,
      grad_fn = x$grad_fn
    ),
    path = path
  )
}
