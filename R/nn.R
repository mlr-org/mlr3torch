#' @title Create a Neural Network Layer
#' @description
#' Retrieve a neural network layer from the
#' [`mlr_pipeops`][mlr3pipelines::mlr_pipeops] dictionary.
#'
#' The id of the returned [`PipeOp`][mlr3pipelines::PipeOp] is `.key`.
#' Because the ids within a [`Graph`][mlr3pipelines::Graph] must be unique, `.key` can be suffixed
#' with `_<n>` to disambiguate repeated layers, e.g. `nn("linear_1")` and `nn("linear_2")`.
#' @param .key (`character(1)`)\cr
#'   The key of the layer in the dictionary, optionally followed by a `_<n>` suffix.
#' @param ... (any)\cr
#'   Additional parameters, constructor arguments or fields.
#' @export
#' @examples
#' po1 = nn("linear", id = "linear")
#' # is the same as:
#' po2 = nn("linear")
#'
#' # the `_<n>` suffix is part of the id, but not of the dictionary key
#' nn("linear_1")$id
nn = function(.key, ...) {
  assert_string(.key)
  args = list(...)
  if (is.null(args$id)) {
    args$id = .key
  }
  # a `_<n>` suffix only disambiguates repeated layers within a Graph and is not part of the key
  invoke(po, .obj = paste0("nn_", sub("_[0-9]+$", "", .key)), .args = args)
}
