#' @title Convenience Function to Retrieve a PipeOpTorch without the Prefix torch_ or nn_
#' @description
#' All classes that inherit from [PipeOpTorch]
#'
#' @name pot
#' @examples
#' po("nn_linear", id = "linear)
#' # is the same as
#' pot("linear")
#'
#' po("torch_optimizer", "adam", id = "adam") is the same as
#' pot("optimizer", "adam")
#'
#' @export
pot = function(.key, ...) {
  assert_string(.key, min.chars = 1L)

  if (grepl("^nn_", .key) || grepl("^torch_", .key)) {
    stopf("You probably wanted po(\"%s\", ...).", .key)
  }

  nn_key = paste0("nn_", .key)
  torch_key = paste0("torch_", .key)
  obj = try(po(torch_key, ...), silent = TRUE)
  if (inherits(obj, "try-error")) {
    obj = try(po(nn_key, ...), silent = TRUE)
  }
  if (inherits(obj, "try-error")) {
    stopf("PipeOp with neither id %s or %s exists.", torch_key, nn_key)
  }

  obj$id = gsub("^(nn_|torch_)", "", obj$id)

  obj
}

#' @rdname pot
#' @export
pots = function(.keys, ...) {
  objs = map(.keys, pot, ...)
  walk(objs, function(obj) obj$id = gsub("^torch_", "", obj$id))
}
