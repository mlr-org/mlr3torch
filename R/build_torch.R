build_torch = function(self, task, network = NULL) {
  p = self$param_set$get_values(tag = "train")

  pars_optim = p[startsWith(names(p), "opt.")]
  names(pars_optim) = gsub("opt.", "", names(pars_optim), fixed = TRUE)

  pars_loss = p[startsWith(names(p), "loss.")]
  names(pars_loss) = gsub("loss.", "", names(pars_loss), fixed = TRUE)

  pars = remove_named(p, c(names(pars_optim), names(pars_loss)))

  optim_name = get_private(self)$.optimizer
  loss_name = get_private(self)$.loss

  optimizer = invoke(get_optimizer(optim_name), .args = pars_optim, params = network$parameters)
  loss_fn = invoke(get_loss(loss_name), .args = pars_loss)

  list(
    network = network,
    optimizer = optimizer,
    loss_fn = loss_fn
  )
}
