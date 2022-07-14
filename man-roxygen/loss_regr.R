#' @section Loss:
#' For regression tasks, the following lossses are supported and can be set during construction
#' using the `loss` argument:
#'
#'  * mse
#'
#' The default is set to "mse". Depending on the loss the constructor arguments of the
#' corresponding loss funcction are dynamically set as parameters of the learner, prefixed by
#' "loss.", see section *Parameters* for an example using the `loss = "mse"`.
#' The loss function cannot be changed after the learner is constructed.
