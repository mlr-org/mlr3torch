#' @param callbacks (`list()` of [`TorchCallback`]s)\cr
#'  The callbacks used during training.
#'  Must have unique ids.
#'  They are executed in the order in which they are provided, unless a callback requests
#'  otherwise via its `$weight`, see section *Ordering* of [`CallbackSet`].
