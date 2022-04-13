#' Dictionary of TorchOps
#' @export
make_paramset_optim = function(optimizer) {
  # TODO: verify all the parameters here that the ranges make sensee
  optim_paramsets$get(optimizer)
}

make_paramset_adam = function() {
  check_betas = function(x) {
    if (test_numeric(x, lower = 0, upper = 1, any.missing = FALSE, len = 2L)) {
      return(TRUE)
    } else {
      return("Parameter betas invalid, must be a numeric vector of length 2 in (0, 1).")
    }
  }
  ps(
    lr = p_dbl(default = 0.001, lower = 0, upper = Inf, tags = c("train", "optimizer")),
    betas = p_uty(default = c(0.9, 0.999), tags = c("train", "optimizer"), custom_check = check_betas),
    eps = p_dbl(default = 1e-08, lower = 1e-16, upper = 1e-4, tags = c("train", "optimizer")),
    weight_decay = p_dbl(default = 0, lower = 0, upper = 1, tags = c("train", "optimizer")),
    amsgrad = p_lgl(default = FALSE, tags = c("train", "optimizer"))
  )
}

make_paramset_sgd = function() {
  ps(
    lr = p_dbl(default = 0.001, lower = 0, tags = c("required", c("train", "optimizer"))),
    momentum = p_dbl(0, 1, default = 0, tags = c("train", "optimizer")),
    dampening = p_dbl(default = 0, lower = 0, upper = 1, tags = c("train", "optimizer")),
    weight_decay = p_dbl(0, 1, default = 0, tags = c("train", "optimizer")),
    nesterov = p_dbl(default = 0, upper = 1, lower = 0, tags = c("train", "optimizer"))
  )
}

make_paramset_rprop = function() {
  check_etas = function(x) {
    if (test_numeric(x, lower = 0, upper = Inf, finite = TRUE, len = 2L)) {
      return(TRUE)
    } else {
      return("Parameter etas invalid, must be a numeric vector of length 2 in (0, Inf).")
    }

  }
  ps(
    lr = p_dbl(default = 0.01, lower = 0, tags = c("train", "optimizer")),
    etas = p_uty(default = c(0.5, 1.2), tags = c("train", "optimizer")),
    step_sizes = p_uty(c(1e-06, 50), tags = c("train", "optimizer"))
  )
}

make_paramset_rmsprop = function() {
  ps(
    lr = p_dbl(default = 0.01, lower = 0, tags = c("train", "optimizer")),
    alpha = p_dbl(default = 0.99, lower = 0, upper = 1, tags = c("train", "optimizer")),
    eps = p_dbl(default = 1e-08, lower = 1e-16, upper = 1e-4, tags = c("train", "optimizer")),
    weight_decay = p_dbl(default = 0, lower = 0, upper = 1, tags = c("train", "optimizer")),
    momentum = p_dbl(default = 0, lower = 0, upper = 1, tags = c("train", "optimizer")),
    centered = p_lgl(tags = c("train", "optimizer"))
  )
}

make_paramset_adagrad = function() {
  ps(
    lr = p_dbl(default = 0.01, lower = 0, tags = c("train", "optimizer")),
    lr_decay = p_dbl(default = 0, lower = 0, upper = 1, tags = c("train", "optimizer")),
    weight_decay = p_dbl(default = 0, lower = 0, upper = 1, tags = c("train", "optimizer")),
    initial_accumulator_value = p_dbl(default = 0, lower = 0, tags = c("train", "optimizer")),
    eps = p_dbl(default = 1e-10, lower = 1e-16, upper = 1e-4, tags = c("train", "optimizer"))
  )
}

make_paramset_adadelta = function() {
  ps(
    lr = p_dbl(default = 1, lower = 0, tags = c("train", "optimizer")),
    rho = p_dbl(default = 0.9, lower = 0, upper = 1, tags = c("train", "optimizer")),
    eps = p_dbl(1e-06, lower = 1e-16, upper = 1e-4, tags = c("train", "optimizer")),
    weight_decay = p_dbl(default = 0, lower = 0, upper = 1, tags = c("train", "optimizer"))
  )
}

make_paramset_lbfgs = function() {
  ps(
    lr = p_dbl(default = 1, lower = 0, tags = c("train", "optimizer")),
    max_iter = p_int(default = 20, lower = 1, tags = c("train", "optimizer")),
    max_eval = p_dbl(default = "max_iter * 1.25", lower = 1L, tags = c("train", "optimizer")),
    tolerance_grad = p_dbl(default = 1e-07, lower = 1e-16, upper = 1e-4, tags = c("train", "optimizer")),
    tolerance_change = p_dvl(default = 1e-09, lower = 1e-16, upper = 1e-4, tags = c("train", "optimizer")),
    history_size = p_int(default = 100L, lower = 1L, tags = c("train", "optimizer")),
    line_search_fn = p_fct(default = NULL, special_vals = list(NULL), levels = list("strong_wolfe"),
      tags = c("train", "optimizer"))
  )
}

optim_paramsets = Dictionary$new()
optim_paramsets$add("adam", make_paramset_adam)
optim_paramsets$add("sgd", make_paramset_sgd)
optim_paramsets$add("rprop", make_paramset_rprop)
optim_paramsets$add("rmsprop", make_paramset_rmsprop)
optim_paramsets$add("adagrad", make_paramset_adagrad)
optim_paramsets$add("adadelta", make_paramset_adadelta)
optim_paramsets$add("lbfgs", make_paramset_lbfgs)
