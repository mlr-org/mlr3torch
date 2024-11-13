#' @title Selector Functions for Neural Network Parameters
#' 
#' @name SelectorParam
#' 
#' @description
#' A [`SelectorParam`] function is used by the callback `CallbackSetUnfreeze` to determine a subset of parameters to freeze or unfreeze during training.
#' 
#' @section Details:
#' ...

make_selectorparam = function(fun, description, ...) {
  structure(fun,
    repr = sprintf(description, ...),
    class = c("SelectorParam", "function")
  )
}

#' @describeIn SelectorParam `selectorparam_all` selects all parameters
#' @export
selectorparam_all = function() {
  make_selectorparam(function(param_names) {
    param_names
  }, "selectorparam_all()")
}

#' @describeIn SelectorParam `selectorparam_none` selects no parameters
#' @export
selectorparam_none = function() {
  make_selectorparam(function(param_names) {
    character(0)
  }, "selectorparam_none()")
}

selectorparam_grep = function(pattern, ignore.case = FALSE, perl = FALSE, fixed = FALSE) {
  assert_character(pattern)
  assert_flag(ignore.case)
  assert_flag(perl)
  assert_flag(fixed)
  str_ignore_case = if (ignore.case) ", ignore.case = TRUE" else ""
  str_perl = if (perl) ", perl = TRUE" else ""
  str_fixed = if (fixed) ", fixed = TRUE" else ""
  make_selectorparam(function(param_names) {
    grep(pattern, param_names, ignore.case = ignore.case, perl = perl, fixed = fixed, value = TRUE)
  }, "selector_grep(%s%s%s%s)", pattern, str_ignore_case, str_perl, str_fixed)
}

selectorparam_name = function(param_names, assert_present = FALSE) {
  assert_character(feature_names, any.missing = FALSE)
  assert_flag(assert_present)
  str_assert_present = if (assert_present) ", assert_present = TRUE" else ""
  make_selectorparam(function(full_names) {
    if (assert_present) {
      assert_subset(param_names, full_names)
    }
  }, "selectorparam_name(%s%s)", char_repr(feature_names), str_assert_present)
}

# selector_invert = function(selector) {
#   assert_function(selector)
#   make_selectorparam(function())
# }