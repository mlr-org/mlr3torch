#' @title Selector Functions for Neural Network Parameters
#' 
#' @name Select
#' 
#' @description
#' A [`Select`] function is used by the callback `CallbackSetUnfreeze` to determine a subset of parameters to freeze or unfreeze during training.
#' 
#' @section Details:
#' ...
NULL

make_select = function(fun, description, ...) {
  structure(fun,
    repr = sprintf(description, ...),
    class = c("Select", "function")
  )
}

#' @describeIn Select `select_all` selects all parameters
#' @export
select_all = function() {
  make_select(function(param_names) {
    param_names
  }, "select_all()")
}

#' @describeIn Select `select_none` selects no parameters
#' @export
select_none = function() {
  make_select(function(param_names) {
    character(0)
  }, "select_none()")
}

#' @describeIn Select `select_grep` selects parameters with names matching a regular expression
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

#' @describeIn Select `select_name` selects parameters with names matching the given names
#' @export
select_name = function(param_names, assert_present = FALSE) {
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

#' @describeIn Select `select_invert` selects the parameters NOT selected by the given selector
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