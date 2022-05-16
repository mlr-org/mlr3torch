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
tops = function(.objs, ...) {
  UseMethod("tops")
}

#' @export
top.character = function(.obj, ...) {
  key = extract_key(.obj)
  if (length(key)) {
    assert_true(!hasArg("id"))
    return(dictionary_sugar_get(dict = mlr_torchops, .key = key, id = .obj, ...))
  }
  dictionary_sugar_get(dict = mlr_torchops, .key = .obj, ...)
}


#' @export
tops.character = function(.objs, ...) {
  map(.objs, top.character, ...)
}

extract_key = function(x) {
  if (grepl("_\\d+$", x)) {
    split = strsplit(x, split = "_")[[1L]]
    key = paste0(split[-length(split)], collapse = "_")
    return(key)
  }

  return(x)
}
