#' @param optimizer ([`TorchOptimizer`])\cr
#'   The optimizer used to train the network.
#' @param loss ([`TorchLoss`])\cr
#'   The loss used to train the network.
#' @param callbacks (`list()` of [`TorchCallback`]s)\cr
#'  The callbacks. Must have unique ids distinct from "history".
