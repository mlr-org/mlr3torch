#' @title Base Class for Torch Descriptors
#'
#' @description
#' Abstract Base Class from which [`TorchLoss`], [`TorchOptimizer`], and [`TorchCallback`] inherit.
#' This class wraps a generator (R6Class Generator or the torch version of such a generator) and annotates it
#' with metadata such as a [`ParamSet`][paradox::ParamSet], a label, an ID, packages, or a manual page.
#'
#' The parameters are the construction arguments of the wrapped generator and the parameter `$values` are passed
#' to the generator when calling the public method `$generate()`.
#'
#' @section Parameters:
#' Defined by the constructor argument `param_set`.
#' All parameters are tagged with `"train"`, but this is done automatically during initialize.
#'
#' @family Torch Descriptor
#' @export
TorchDescriptor = R6Class("TorchDescriptor",
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
    initialize = function(generator, id = NULL, param_set = NULL, packages = NULL, label = NULL, man = NULL) {
      assert_true(is.function(generator) || inherits(generator, "R6ClassGenerator"))
      self$generator = generator
      self$param_set = assert_r6(param_set, "ParamSet", null.ok = TRUE) %??% inferps(generator)
      if (length(self$param_set$tags)) {
        self$param_set$tags = map(self$param_set$tags, function(tags) union(tags, "train"))

      }
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
      self$id = assert_string(id %??% class(generator)[[1L]], min.chars = 1L)
      self$label = assert_string(label %??% self$id, min.chars = 1L)
      self$packages = assert_names(unique(union(packages, "torch")), type = "strict")
    },
    #' @description
    #' Prints the object
    #' @param ... any
    print = function(...)  {
      repr = if (test_class(self$generator, "R6ClassGenerator")) {
        self$generator$classname
      } else {
        class(self$generator)[[1L]]
      }
      catn(sprintf("<%s:%s> %s", class(self)[[1L]], self$id, self$label))
      catn(str_indent("* Generator:", repr))
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
  active = list(
    #' @template field_phash
    phash = function() {
      # This phash is only heuristic but should realistically always work.
      calculate_hash(class(self), self$id, self$packages, self$label, self$man, self$param_set$ids(),
        self$param_set$class, class(self$generator), private$.additional_phash_input()
      )
    }
  ),
  private = list(
    .additional_phash_input = function() {
      stopf("Classes inheriting from TorchDescriptor must implement the .additional_phash_input() method.")
    },
    deep_clone = function(name, value) {
      if (name == "generator") {
        return(value)
      } else if (is.R6(value)) {
        value$clone(deep = TRUE)
      } else {
        value
      }
    }
  )
)
