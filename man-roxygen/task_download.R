#' @section Download:
#' The [task][Task]'s backend is a [`DataBackendLazy`] which will download the data once it is requested.
#' Other meta-data is already available before that.
#' You can cache these datasets by setting the `mlr3torch.cache` option to `TRUE` or to a specific path to be used
#' as the cache directory.
