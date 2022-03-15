#' @title Shorthand TorchOp Constructor
#'
#' @description
#' Create
#'  - a `TorchOp` from `mlr_torchops` from given ID
#'
#' @export
top = function(.obj, ...) {
  UseMethod("top")
}

#' @rdname top
#' @export
tops = function(.ob, ...) {
  UseMethod("tops")
}

#' @export
top.character = function(.obj, ...) {
  dictionary_sugar_get(dict = mlr_torchops, .key = .obj, ...)
}

#' @export
tops.character = function(.objs, ...) {
  dictionary_sugar_mget(dict = mlr_torchops, .keys = .objs, ...)
}
