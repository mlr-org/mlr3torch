#' @title Convenience Function to Retrieve a PipeOpTorch without the Prefix
#'
#' @description
#' Sugar function that allows to construct all [PipeOp]s relating to torch without needing to
#' write out the the prefixes `"torch_"` and `"nn_"`.
#' The `id` of the resulting object is also without the prefix.
#'
#' Possible objects to construct are:
#' * All objects inheriting from [PipeOpTorch], i.e. [PipeOpTorchLinear], [PipeOpTorchReLU], etc..
#' * All objects inheritng from [PipeOpTorchIngress] which is the entry point to the network.
#' * [PipeOpTorchLoss] which allows to configure the loss function.
#' * [PipeOpTorchOptimizer] which allows to configure the loss Optimizer.
#' * All objects inheriting from [PipeOpTorchModel] which transforms the [ModelDescriptor] into
#'   a torch [mlr3::Learner] and trains it.
#'
#' @name pot
#' @examples
#' po("nn_linear", id = "linear")
#' # is the same as
#' pot("linear")
#'
#' po("torch_optimizer", "adam", id = "adam")
#' # is the same as
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
  walk(objs, function(obj) obj$id = gsub("^(nn_|torch_)", "", obj$id))
}
