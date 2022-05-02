#' Dictionary of TorchOps
#' @export
make_paramset_optim = function(optimizer) {
  # TODO: verify all the parameters here that the ranges make sensee
  paramsets_optim$get(optimizer)
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

make_paramset_asgd = function() {
  ps(
    lr = p_dbl(default = 1e-2, lower = 0, tags = c("required", c("train", "optimizer"))),
    lambd = p_dbl(lower = 0, upper = 1, default = 1e-4, tags = c("train", "optimizer")),
    alpha = p_dbl(lower = 0, upper = Inf, default = 0.75, tags = c("train", "optimizer")),
    t0 = p_int(lower = 1L, upper = Inf, default = 1e6, tags = c("train", "optimizer")),
    weight_decay = p_dbl(default = 0, lower = 0, upper = 1, tags = c("train", "optimizer"))
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
    eps = p_dbl(default = 1e-06, lower = 1e-16, upper = 1e-4, tags = c("train", "optimizer")),
    weight_decay = p_dbl(default = 0, lower = 0, upper = 1, tags = c("train", "optimizer"))
  )
}

make_paramset_lbfgs = function() {
  ps(
    lr = p_dbl(default = 1, lower = 0, tags = c("train", "optimizer")),
    max_iter = p_int(default = 20, lower = 1, tags = c("train", "optimizer")),
    max_eval = p_dbl(lower = 1L, tags = c("train", "optimizer")),
    tolerance_grad = p_dbl(default = 1e-07, lower = 1e-16, upper = 1e-4, tags = c("train", "optimizer")),
    tolerance_change = p_dbl(default = 1e-09, lower = 1e-16, upper = 1e-4, tags = c("train", "optimizer")),
    history_size = p_int(default = 100L, lower = 1L, tags = c("train", "optimizer")),
    line_search_fn = p_fct(default = "strong_wolfe", levels = c("strong_wolfe"),
      tags = c("train", "optimizer"))
  )
}

paramsets_optim = Dictionary$new()
paramsets_optim$add("adam", make_paramset_adam)
paramsets_optim$add("sgd", make_paramset_sgd)
paramsets_optim$add("asgd", make_paramset_asgd)
paramsets_optim$add("rprop", make_paramset_rprop)
paramsets_optim$add("rmsprop", make_paramset_rmsprop)
paramsets_optim$add("adagrad", make_paramset_adagrad)
paramsets_optim$add("adadelta", make_paramset_adadelta)
paramsets_optim$add("lbfgs", make_paramset_lbfgs)
