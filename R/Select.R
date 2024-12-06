#' @title Selector Functions for Character Vectors
#'
#' @name Select
#'
#' @description
#' A [`Select`] function subsets a character vector. They are used by the callback `CallbackSetUnfreeze` to select parameters to freeze or unfreeze during training.
#' ...
NULL

make_select = function(fun, description, ...) {
  structure(fun,
    repr = sprintf(description, ...),
    class = c("Select", "function")
  )
}

#' @describeIn Select `select_all` selects all elements
#' @export
select_all = function() {
  make_select(function(param_names) {
    param_names
  }, "select_all()")
}

#' @describeIn Select `select_none` selects no elements
#' @export
select_none = function() {
  make_select(function(param_names) {
    character(0)
  }, "select_none()")
}

#' @describeIn Select `select_grep` selects elements with names matching a regular expression
#' @param pattern See `grep()`
#' @param ignore.case See `grep()`
#' @param perl See `grep()`
#' @param fixed See `grep()`
#' @export
select_grep = function(pattern, ignore.case = FALSE, perl = FALSE, fixed = FALSE) {
  assert_character(pattern)
  assert_flag(ignore.case)
  assert_flag(perl)
  assert_flag(fixed)
  str_ignore_case = if (ignore.case) ", ignore.case = TRUE" else ""
  str_perl = if (perl) ", perl = TRUE" else ""
  str_fixed = if (fixed) ", fixed = TRUE" else ""
  make_select(function(param_names) {
    grep(pattern, param_names, ignore.case = ignore.case, perl = perl, fixed = fixed, value = TRUE)
  }, "selector_grep(%s%s%s%s)", pattern, str_ignore_case, str_perl, str_fixed)
}

#' @describeIn Select `select_name` selects elements with names matching the given names
#' @param param_names The names of the parameters that you want to select
#' @param assert_present Whether to check that `param_names` is a subset of the full vector of names
#' @export
select_name = function(param_names, assert_present = TRUE) {
  assert_character(param_names, any.missing = FALSE)
  assert_flag(assert_present)
  str_assert_present = if (assert_present) ", assert_present = TRUE" else ""
  make_select(function(full_names) {
    if (assert_present) {
      assert_subset(param_names, full_names)
    }
    intersect(full_names, param_names)
  }, "select_name(%s%s)", char_repr(param_names), str_assert_present)
}

#' @describeIn Select `select_invert` selects the elements NOT selected by the given selector
#' @param select A `Select`
#' @export
select_invert = function(select) {
  assert_function(select)
  make_select(function(full_names) {
    setdiff(full_names, select(full_names))
  }, "select_invert(%s)", select_repr(select))
}

# copied from mlr3pipelines
# Representation of character vector
# letters[1]   --> '"a"'
# letters[1:2] --> 'c("a", "b")'
char_repr = function(x) {
  output = str_collapse(x, sep = ", ", quote = '"')
  if (length(x) == 0) {
    "character(0)"
  } else if (length(x) == 1) {
    output
  } else {
    sprintf("c(%s)", output)
  }
}

# copied from mlr3pipelines
# Representation for a function that may or may not be a `Select`.
# If it is not, we just use deparse(), otherwise we use the repr as
# reported by that selector.
select_repr = function(select) {
  if (test_string(attr(select, "repr"))) {
    attr(select, "repr")
  } else {
    str_collapse(deparse(select), sep = "\n")
  }
}

# copied from mlr3pipelines
#' @export
print.Select = function(x, ...) {
  if (inherits(x, "R6")) return(NextMethod("print"))
  cat(paste0(attr(x, "repr"), "\n"))
  invisible(x)
}
