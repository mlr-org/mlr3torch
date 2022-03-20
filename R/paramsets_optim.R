#' TODO: verify all the parameters here that the ranges make sensee
make_paramset_optim = function(optimizer) {
  get(sprintf("make_paramset_%s", optimizer))()
}

make_paramset_adam = function() {
  ps(
    lr = p_dbl(default = 0.001, min = 0, tags = "train"),
    betas = p_uty(default = c(0.9, 0.999), tags = "train"),
    eps = p_dbl(1e-08, tags = "train"),
    weight_decay = p_dbl(0, 1, default = 0, tags = "train"),
    amsgrad = p_lgl(default = FALSE, tags = "train")
  )
}

make_paramset_sgd = function() {
  ps(
    lr = p_dbl(default = 0.001, min = 0, tags = c("required", "train")),
    momentum = p_dbl(0, 1, default = 0, tags = "train"),
    weight_decay = p_dbl(0, 1, default = 0, tags = "train"),
    nesterov = p_lgl(0, 1, default = 0, tags = "train"),
  )
}

# make_paramset_rmsprop = function() {
#   ps(
#     lr = p_dbl(0, default = 0.01, tags = "train"),
#     alpha = p_dbl(0, )
#
#   )
# }
