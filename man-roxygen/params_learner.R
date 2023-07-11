#' @param task_type (`character(1)`)\cr
#'   The task type, either `"classif`" or `"regr"`.
#' @param optimizer ([`TorchOptimizer`])\cr
#'   The optimizer to use for training.
#'   Per default, *adam* is used.
#' @param loss ([`TorchLoss`])\cr
#'   The loss used to train the network.
#'   Per default, *mse* is used for regression and *cross_entropy* for classification.
#' @param callbacks (`list()` of [`TorchCallback`]s)\cr
#'  The callbacks. Must have unique ids.
