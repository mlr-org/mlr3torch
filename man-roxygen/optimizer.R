#' @section Optimizer:
#' The following optimizers are supported and can be set during construction using the
#' `optimizer` argument.
#'
#'  * adadelta
#'  * adagrad
#'  * adam
#'  * asgd
#'  * lbfgs
#'  * rmsprop
#'  * rprop
#'  * sgd
#'
#' Depending on `optimizer`, the constructor arguments of the corresponding
#' [optimizer][torch::optimizer] are dynamically set as parameters of the learner, prefixed by
#' "opt.".
#' The optimizer cannot be changed after the learner was constructed.
