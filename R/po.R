#' @title Shorthand TorchOp Constructor
#'
#' @description
#' Create
#'  - a `TorchOp` from `mlr_torchops` from given ID
#'
#' @export
to = function(.obj, ...) {
  UseMethod("to")
}

#' @rdname po
#' @export
tos = function(.ob, ...) {
  UseMethod("tos")
}

to.character = function(.obj, ...) {
  dictionary_sugar_get(dict = mlr_pipeops, .key = .obj, ...)
}

tos.character = function(.objs, ...) {
  dictionary_sugar_mget(dict = mlr_torchops, .keys = .objs, ...)
}
