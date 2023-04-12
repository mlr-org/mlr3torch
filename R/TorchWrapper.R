#' @title Base Class for Torch Wrappers
#' @usage NULL
#' @name torch_wrapper
#' @format `r roxy_format(TorchWrapper)`
#'
#' @description
#' Abstract Base Class from which [`TorchLoss`], [`TorchOptimizer`], and [`TorchCallback`] inherit.
#' This class wraps a generator (R6Class Generator or the torch version of such a generator) and annotates it.
#'
#' @section Construction:
#' `r roxy_construction(TorchWrapper)`
#' * `generator` :: `function` or [`R6ClassGenerator`]\cr
#'   The wrapped generator that is described.
#' * `id` :: `character(1)`\cr
#'   The identifier of the object. Used to e.g. retrieve it from a dictionary.
#' * `param_set` :: [`paradox::ParamSet`]\cr
#'   The parameter set that describes the arguments of the generator.
#' * `packages` :: `character()`\cr
#'   The packages the generator depends on.
#' * `label` :: `character(1)`\cr
#'   The label, which is used for printing. Defaults to `id`.
#' * `man` :: (`character(1)`)\cr
#'   String in the format `[pkg]::[topic]` pointing to a manual page for this object.
#' @param man (`character(1)`)\cr
#'   String in the format `[pkg]::[topic]` pointing to a manual page for this object.
#'   The referenced help package can be opened via method `$help()`.
#'
#' @section Parameters:
#' Defined by the constructor argument `param_set`.
#'
#' @section Fields:
#' * `generator` :: `function` or [`R6ClassGenerator`]\cr
#'   The wrapped generator that is described.
#' * `id` :: `character(1)`\cr
#'   The identifier of the object. Used to e.g. retrieve it from a dictionary.
#' * `param_set` :: [`paradox::ParamSet`]\cr
#'   The parameter set that describes the arguments of the generator.
#' * `packages` :: `character()`\cr
#'   The packages the generator depends on.
#' * `label` :: `character(1)`\cr
#'   The label, which is used for printing. Defaults to `id`.
#' * `man` :: (`character(1)`)\cr
#'   String in the format `[pkg]::[topic]` pointing to a manual page for this object.
#'   The referenced help package can be opened via method `$help()`.
#' @section Methods:
#' * `generate()`\cr
#'    () -> any
#'    Calls the generator with the given parameter values.
#' * `print(...)`\cr
#'    () -> `CallbackTorch`
#'    Prints the object.
#' * `help()`\cr
#'    () -> help file\cr
#'    Displays the help file.
#' @family torch_wrappers
#' @export
TorchWrapper = R6Class("TorchWrapper",
  public = list(
    label = NULL,
    param_set = NULL,
    packages = NULL,
    id = NULL,
    generator = NULL,
    man = NULL,
    initialize = function(generator, id, param_set = NULL, packages = NULL, label = id, man = NULL) {
      assert_true(is.function(generator) || inherits(generator, "R6ClassGenerator"))
      self$generator = generator
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
    print = function()  {
      catn(sprintf("<%s:%s> %s", class(self)[[1L]], self$id, self$label))
      catn(str_indent("* Generator:", private$.repr))
      catn(str_indent("* Parameters:", as_short_string(self$param_set$values, 1000L)))
      catn(str_indent("* Packages:", as_short_string(self$packages, 1000L)))
      invisible(self)
    },
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
    help = function() {
      open_help(self$man)
    }
  ), 
  private = list(
    .repr = NULL
  )
)
