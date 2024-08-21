#' @title Create a Neural Network Layer
#' @description
#' Retrieve a neural network layer from the
#' [`mlr_pipeops`][mlr3pipelines::mlr_pipeops] dictionary.
#' @param .key (`character(1)`)\cr
#' @param ... (any)\cr
#'   Additional parameters, constructor arguments or fields.
#' @export
#' @examples
#' po1 = po("nn_linear", id = "linear")
#' # is the same as:
#' po2 = nn("linear")
nn = function(.key, ...) {
  invoke(po, .obj = paste0("nn_", .key), .args = insert_named(list(id = .key), list(...)))
}
