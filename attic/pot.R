#' @title Convenience Function to Retrieve a PipeOp without the "nn" or "torch" Prefix
#'
#' @description
#' Sugar function that allows to construct all [PipeOp]s relating to torch without needing to
#' write out the the prefixes `"torch_"` and `"nn_"`.
#' Note that the assigned id has the prefix.
#'
#' @name pot
#' @export
#' @examples
#' po("nn_linear", id = "linear")
#' # is the same as
#' pot("linear")
#'
#' po("torch_optimizer", "adam", id = "adam")
#' # is the same as
#' pot("optimizer", "adam")
#'
pot = function(.key, ...) {
  assert_string(.key, min.chars = 1L)
  if (grepl("^nn_", .key) || grepl("^torch_", .key)) {
    stopf("You probably wanted po(\"%s\", ...).", .key)
  }
  hits = 0
  torch_key = paste0("torch_", .key)
  nn_key = paste0("nn_", .key)
  if (torch_key %in% mlr_pipeops$keys()) {
    hits = hits + 1
    key = torch_key
  }
  if (nn_key %in% mlr_pipeops$keys()) {
    hits = hits + 1
    key = nn_key
  }
  if (hits == 0) {
    stopf("PipeOp with neither id %s or %s exists.", torch_key, nn_key)
  } else if (hits == 2) {
    stopf("PipeOp with both id %s or %s exists, which is ambiguous, use `po(...)` instead.", torch_key, nn_key)
  }
  po(key, ...)
}

#' @rdname pot
#' @export
pots = function(.keys, ...) {
  objs = map(.keys, pot, ...)
}
