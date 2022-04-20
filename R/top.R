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
  chars = strsplit(x, "")[[1L]]
  is_digit = suppressWarnings(!is.na(as.numeric(chars)))
  if (!sum(is_digit)) {
    return(character(0))
  }
  ids = which(is_digit)
  mx = max(ids)
  mn = min(ids)
  l = length(chars)
  n = sum(is_digit)
  # checks that the last values are continously digits, abc123 is valid, ab1c23 is invalid
  assert_true((mx == l) && (mn == (l - n + 1)))
  return(paste0(chars[!is_digit], collapse = ""))
}
