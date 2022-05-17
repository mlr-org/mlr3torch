#' @title Shorthand TorchOp Constructor
#'
#' @description
#' Create a `TorchOp` from `mlr_torchops` from given ID.
#' The object is initialized with given parameters and `param_vals`.
#'
#' @param .obj (`character(1)`)\cr
#'   The object from which to construct a `PipeOp`. If this is a
#'   `character(1)`, it is looked up in the [`mlr_pipeops`] dictionary.
#'   Otherwise, it is converted to a `PipeOp`.
#'   It is possible to append a `"_[number]"` to the id, to conveniently extract the element
#'   specified by the id without the suffix, while setting giving a different id to the `TorchOp`.
#'   []
#' @param ... `any`\cr
#'   Additional parameters to give to constructed object.
#'   This may be an argument of the constructor of the
#'   `TorchOp`, in which case it is given to this constructor;
#'   or it may be a parameter value, in which case it is
#'   given to the `param_vals` argument of the constructor.
#' @export
#' @examples
#' # these are all equivalent
#' top("linear", out_features = 10L, id = "linear_1")
#' top("linear_1", out_features = 10L)
#' mlr_torchops$get("linear", out_features = 10L, id = "linear_1")
#'
#' @export
top = function(.obj, ...) {
  key = extract_key(.obj)
  dictionary_sugar_get(dict = mlr_torchops, .key = .obj, ...)
}

extract_key = function(x) {
  if (grepl("_\\d+$", x)) {
    split = strsplit(x, split = "_")[[1L]]
    key = paste0(split[-length(split)], collapse = "_")
    return(key)
  }

  return(x)
}
