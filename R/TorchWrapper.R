#' @title Base Class for Torch Wrappers
#' @name torch_wrapper
#'
#' @description
#' Abstract Base Class from which [`TorchLoss`], [`TorchOptimizer`], and [`TorchCallback`] inherit.
#' This class wraps a generator (R6Class Generator or the torch version of such a generator) and annotates it
#' with metadata such as a [`ParamSet`], a label, an ID, packages, or a manual page.
#'
#' The parameters are the construction arguments of the wrapped generator and the parameter `$values` are passed
#' to the generator when calling the public method `$generate()`.
#'
#' @section Parameters:
#' Defined by the constructor argument `param_set`.
#'
#' @family Torch Wrapper
#' @export
TorchWrapper = R6Class("TorchWrapper",
  public = list(
    #' @template field_label
    label = NULL,
    #' @template field_param_set
    param_set = NULL,
    #' @template field_packages
    packages = NULL,
    #' @template field_id
    id = NULL,
    #' @field generator
    #'   The wrapped generator that is described.
    generator = NULL,
    #' @template field_man
    man = NULL,
    #' @description Creates a new instance of this [R6][R6::R6Class] class.
    #' @template param_id
    #' @template param_param_set
    #' @param generator
    #'   The wrapped generator that is described.
    #' @template param_packages
    #' @template param_label
    #' @template param_man
    initialize = function(generator, id, param_set = NULL, packages = NULL, label = id, man = NULL) {
      assert_true(is.function(generator) || inherits(generator, "R6ClassGenerator"))
      self$generator = generator
      # TODO: Assert that all parameters are tagged with "train"
      self$param_set = assert_r6(param_set, "ParamSet", null.ok = TRUE) %??% inferps(generator)
      if (is.function(generator)) {
        args = formalArgs(generator)
      } else {
        init = get_init(generator)
        if (is.null(init)) {
          args = c()
        } else {
          args = formalArgs(init)
        }
      }
      if ("..." %nin% args && !test_subset(self$param_set$ids(), args)) {
        missing = setdiff(self$param_set$ids(), args)
        stopf("Parameter values with ids %s are missing in generator.", paste0("'", missing, "'", collapse = ", "))
      }
      self$man = assert_string(man, null.ok = TRUE)
      self$id = assert_string(id, min.chars = 1L)
      self$label = assert_string(label, min.chars = 1L)
      self$packages = assert_names(unique(union(packages, c("torch", "mlr3torch"))), type = "strict")

      private$.repr = if (test_class(self$generator, "R6ClassGenerator")) {
        self$generator$classname
      } else {
        class(self$generator)[[1L]]
      }
    },
    #' @description
    #' Prints the object
    #' @param ... any
    print = function(...)  {
      catn(sprintf("<%s:%s> %s", class(self)[[1L]], self$id, self$label))
      catn(str_indent("* Generator:", private$.repr))
      catn(str_indent("* Parameters:", as_short_string(self$param_set$values, 1000L)))
      catn(str_indent("* Packages:", as_short_string(self$packages, 1000L)))
      invisible(self)
    },
    #' @description
    #' Calls the generator with the given parameter values.
    generate = function() {
      require_namespaces(self$packages)
      # The torch generators could also be constructed with the $new() method, but then the return value
      # would be the actual R6 class and not the wrapped function.
      if (is.function(self$generator)) {
        invoke(self$generator, .args = self$param_set$get_values())
      } else {
        invoke(self$generator$new, .args = self$param_set$get_values())
      }
    },
    #' @description
    #'    Displays the help file of the wrapped object.
    help = function() {
      open_help(self$man)
    }
  ),
  private = list(
    .repr = NULL
  )
)
