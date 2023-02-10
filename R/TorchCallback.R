#' @title Convert to a [`TorchCallback`]
#' @description
#' Converts an object to a [`TorchCallback`].
#'
#' @param x (any)\cr
#'   Object to be converted/
#' @param clone (`logical(1)`\cr
#'   Whether to make a deep clone.
#' @param ... (any)\cr
#'   Additional arguments
#' @param param_set ([`ParamSet`])\cr
#'   The parameter set.
#'
#' @return [`TorchCallback`].
#'
#' @export
as_torch_callback = function(x, clone = FALSE, ...) {
  assert_flag(clone)
  UseMethod("as_torch_callback")
}

#' @export
as_torch_callback.TorchCallback = function(x, clone = FALSE) { # nolint
  if (clone) x$clone(deep = TRUE) else x
}

#' @export
as_torch_callback.R6ClassGenerator = function(x, clone = FALSE, param_set = NULL, ...) { # nolint
  # no need to clone because we have a class generator
  # TorchCallback$new(callback_generator = x, param_set = param_set$clone(deep = TRUE), ...)
  if (!is.null(param_set)) {
    assert_param_set(param_set)
    param_set = if (clone) param_set$clone(deep = TRUE) else param_set
  }
  TorchCallback$new(callback_generator = x, param_set = param_set)
}

#' @export
as_torch_callback.character = function(x, clone = FALSE, ...) { # nolint
  t_clbk(.key = x, ...)
}

#' @title Convert to a list of Torch Callbacks
#' @description
#' Converts an object to a list of [`TorchCallback`].
#'
#' @param x (any)\cr
#'   Object to convert.
#' @param clone (`logical(1)`)\cr
#'   Whether to create a deep clone.
#' @param ... (any)\cr
#'   Additional arguments.
#'
#' @export
as_torch_callbacks = function(x, clone, ...) {
  UseMethod("as_torch_callbacks")
}

#' @export
as_torch_callbacks.list = function(x, clone = FALSE, ...) { # nolint
  lapply(x, as_torch_callback, clone = clone, ...)
}

#' @export
as_torch_callbacks.default = function(x, clone = FALSE, ...) { # nolint
  list(as_torch_callback(x, clone = clone, ...))
}




#' @title Torch Callback
#'
#' @usage NULL
#' @name torch_callback
#' @format `r roxy_format(TorchCallback)`
#'
#' @description
#' Leight-weight wrapper around Torch Callbacks: A [`TorchCallback`] wraps a [`CallbackTorch`].
#' To conveniently retrieve a [`TorchCallback`], use [`t_clbk`].
#' It is an analogous construct to the classes [`TorchOptimizer`] or [`TorchLoss`].
#'
#' @section Construction: `r roxy_construction(TorchCallback)`
#'
#' * `callback_generator` :: [`R6ClassGenerator`]\cr
#'   The class generator for the callback that is being wrapped.
#' * `param_set` :: [`ParamSet`]\cr
#'   The parameter set of the callback. These values are passed as construction arguments to the wrapped
#'   [`CallbackTorch`]. The default is `NULL` in which case the parameter set is inferred from the construction
#'   arguments of the wrapped callbacks.
#' * `packages` :: `character()`\cr
#'   The packages the callback depends on.
#'
#' @section Fields:
#' * `callback` :: [`R6ClassGenerator`]\cr
#'   The class generator for the R6 callback.
#' * `param_set` :: [`ParamSet`]\cr
#'   The parameter set. Its values are passed to `$get_callback()`.
#'   Note that the `param_set` is not cloned, so this has to be done before by the user.
#' * `id` :: `character(1)`/cr
#'   The identifier of the callback. This is equal to the identifier of the wrapped [`CallbackTorch`].
#'
#' @section Methods:
#' * `get_callback()`\cr
#'    () -> `CallbackTorch`
#'    Initializes an instance of the class of the wrapped callback with the given parameter specification.
#'
#' @family torch_wrapper
#' @include utils.R
#' @export
TorchCallback = R6Class("TorchCallback",
  public = list(
    initialize = function(callback_generator, param_set = NULL, packages = NULL) {
      private$.callback = assert_class(callback_generator, "R6ClassGenerator")
      private$.packages = assert_character(union(packages, "mlr3torch"), any.missing = FALSE)

      id = callback_generator$public_fields$id
      if (is.null(id)) {
        stopf("Callback generator must have public field 'id'.")
      } else {
        private$.id = id
      }

      init_method = get_init(callback_generator)
      if (is.null(param_set)) {
        if (!is.null(init_method)) {
          private$.param_set = inferps(init_method, tags = character(0))
        } else {
          private$.param_set = ps()
        }
      } else {
        private$.param_set = assert_param_set(param_set)
        if (is.null(init_method)) {
          assert_true(param_set$length == 0)
        } else {
          assert_set_equal(param_set$ids(), formalArgs(init_method))
        }
      }
    },
    print = function(...) {
      catn(sprintf("<TorchCallback: %s>", self$callback$public_fields$id))
      catn(str_indent("* Generator:", self$callback$classname))
      catn(str_indent("* Parameters:", as_short_string(self$param_set$values, 1000L)))

    },
    get_callback = function() {
      require_namespaces(self$packages)
      invoke(self$callback$new, .args = self$param_set$get_values())
    }
  ),
  active = list(
    id = function(rhs) {
      assert_ro_binding(rhs)
      private$.id
    },
    callback = function(rhs) {
      assert_ro_binding(rhs)
      private$.callback
    },
    param_set = function(rhs) {
      assert_ro_binding(rhs)
      private$.param_set
    },
    packages = function(rhs) {
      assert_ro_binding(rhs)
      private$.packages
    }
  ),
  private = list(
    .callback = NULL,
    .param_set = NULL,
    .id = NULL,
    .packages = NULL
  )
)
