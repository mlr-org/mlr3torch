#' Get available devices for torch
#'
#' If no GPU is available
#' @return Character of length `torch::cuda_device_count() + 1`.
#' "cpu" if no GPU is available, otherwise "cpu" and 0-indexed GPUs prefixed
#' with "cuda:".
#' @export
#'
#' @examples
#' # Returns at least "cpu"
#' get_available_device()
get_available_device <- function() {
  devices <- NULL

  if (torch::cuda_is_available()) {
    devices <- paste0("cuda:", seq(0, torch::cuda_device_count() - 1))
  }

  c("cpu", devices)
}

#' Choose the correct device for torch
#'
#' @param gpu_index `[0]` Preferred index (starting from 0) of GPU to use.
#' Will be checked against `get_available_device()`
#'
#' @return `character(1)` with a valid torch device.
#' @export
#'
#' @examples
#' # GPU 0 or CPU
#' choose_device()
#'
#' # Choose second GPU if available
#' # choose_device(1)
choose_device <- function(gpu_index = 0) {
  checkmate::assert_int(gpu_index)

  if (!torch::cuda_is_available()) return("cpu")

  device <- paste0("cuda:", gpu_index)

  stopifnot(device %in% get_available_device())

  device
}

#' Check if provided device is actually available
#' Used as custom_check in ParamUty
#' @keywords internals
#' @param x Parameter passed for `device` in ParamUty.
param_check_device <- function(x) {
  x %in% get_available_device()
}
