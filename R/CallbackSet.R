#' @title Base Class for Callbacks
#'
#' @name mlr_callback_set
#'
#' @description
#' Base class from which callbacks should inherit (see section *Inheriting*).
#' A callback set is a collection of functions that are executed at different stages of the training loop.
#' They can be used to gain more control over the training process of a neural network without
#' having to write everything from scratch.
#'
#' When used a in torch learner, the `CallbackSet` is wrapped in a [`TorchCallback`].
#' The latters parameter set represents the arguments of the [`CallbackSet`]'s `$initialize()` method.
#'
#' @section Inheriting:
#' For each available stage (see section *Stages*) a public method `$on_<stage>()` can be defined.
#' The evaluation context (a [`ContextTorch`]) can be accessed via `self$ctx`, which contains
#' the current state of the training loop.
#' This context is assigned at the beginning of the training loop and removed afterwards.
#' Different stages of a callback can communicate with each other by assigning values to `$self`.
#'
#' *State*:
#' To be able to store information in the `$model` slot of a [`LearnerTorch`], callbacks support a state API.
#' You can overload the `$state_dict()` public method to define what will be stored in `learner$model$callbacks$<id>`
#' after training finishes.
#' This then also requires to implement a `$load_state_dict(state_dict)` method that defines how to load a previously saved
#' callback state into a different callback.
#' Note that the `$state_dict()` should not include the parameter values that were used to initialize the callback.
#'
#' For creating custom callbacks, the function [`torch_callback()`] is recommended, which creates a
#' `CallbackSet` and then wraps it in a [`TorchCallback`].
#' To create a `CallbackSet` the convenience function [`callback_set()`] can be used.
#' These functions perform checks such as that the stages are not accidentally misspelled.
#'
#'
#' @section Stages:
#' * `begin` :: Run before the training loop begins.
#' * `epoch_begin` :: Run he beginning of each epoch.
#' * `batch_begin` :: Run before the forward call.
#' * `after_backward` :: Run after the backward call.
#' * `batch_end` :: Run after the optimizer step.
#' * `batch_valid_begin` :: Run before the forward call in the validation loop.
#' * `batch_valid_end` :: Run after the forward call in the validation loop.
#' * `valid_end` :: Run at the end of validation.
#' * `epoch_end` :: Run at the end of each epoch.
#' * `end` :: Run after last epoch.
#' * `exit` :: Run at last, using `on.exit()`.
#'
#' @section Terminate Training:
#' If training is to be stopped, it is possible to set the field `$terminate` of [`ContextTorch`].
#' At the end of every epoch this field is checked and if it is `TRUE`, training stops.
#' This can for example be used to implement custom early stopping.
#'
#' @family Callback
#' @export
CallbackSet = R6Class("CallbackSet",
  lock_objects = FALSE,
  public = list(
    #' @field ctx ([`ContextTorch`] or `NULL`)\cr
    #'   The evaluation context for the callback.
    #'   This field should always be `NULL` except during the `$train()` call of the torch learner.
    ctx = NULL,
    #' @description
    #' Prints the object.
    #' @param ... (any)\cr
    #'   Currently unused.
    print = function(...) {
      catn(sprintf("<%s>", class(self)[[1L]]))
      catn(str_indent("* Stages:", self$stages))
    },
    #' @description
    #' Returns information that is kept in the the [`LearnerTorch`]'s state after training.
    #' This information should be loadable into the callback using `$load_state_dict()` to be able to continue training.
    #' This returns `NULL` by default.
    state_dict = function() {
      NULL
    },
    #' @description
    #' Loads the state dict into the callback to continue training.
    #' @param state_dict (any)\cr
    #'   The state dict as retrieved via `$state_dict()`.
    load_state_dict = function(state_dict) {
      assert_true(is.null(state_dict))
      NULL
    }
  ),
  active = list(
    #' @field stages (`character()`)\cr
    #'   The active stages of this callback set.
    stages = function(rhs) {
      assert_ro_binding(rhs)
      if (is.null(private$.stages)) {
        private$.stages = mlr_reflections$torch$callback_stages[
          map_lgl(mlr_reflections$torch$callback_stages, function(stage) exists(stage, self, inherits = FALSE))]
      }

      private$.stages
    }

  ),
  private = list(
    .stages = NULL,
    deep_clone = function(name, value) {
      if (name == "ctx" && !is.null(value)) {
        stopf("CallbackSet instances can only be cloned when the 'ctx' is NULL.")
      } else if (is.R6(value)) {
        value$clone(deep = TRUE)
      } else if (is.data.table(value)) {
        copy(value)
      } else {
        value
      }
    }
  )
)

#' @title Create a Set of Callbacks for Torch
#'
#' @description
#' Creates an `R6ClassGenerator` inheriting from [`CallbackSet`].
#' Additionally performs checks such as that the stages are not accidentally misspelled.
#' To create a [`TorchCallback`] use [`torch_callback()`].
#'
#' In order for the resulting class to be cloneable, the private method `$deep_clone()` must be
#' provided.
#'
#' @param classname (`character(1)`)\cr
#'   The class name.
#' @param on_begin,on_end,on_epoch_begin,on_before_valid,on_epoch_end,on_batch_begin,on_batch_end,on_after_backward,on_batch_valid_begin,on_batch_valid_end,on_valid_end,on_exit (`function`)\cr
#'   Function to execute at the given stage, see section *Stages*.
#' @param initialize (`function()`)\cr
#'   The initialization method of the callback.
#' @param public,private,active (`list()`)\cr
#'   Additional public, private, and active fields to add to the callback.
#' @param parent_env (`environment()`)\cr
#'   The parent environment for the [`R6Class`][R6::R6Class].
#' @param inherit (`R6ClassGenerator`)\cr
#'   From which class to inherit.
#'   This class must either be [`CallbackSet`] (default) or inherit from it.
#' @param state_dict (`function()`)\cr
#'   The function that retrieves the state dict from the callback.
#'   This is what will be available in the learner after training.
#' @param load_state_dict (`function(state_dict)`)\cr
#'   Function that loads a callback state.
#' @param lock_objects (`logical(1)`)\cr
#'  Whether to lock the objects of the resulting [`R6Class`][R6::R6Class].
#'  If `FALSE` (default), values can be freely assigned to `self` without declaring them in the
#'  class definition.
#' @family Callback
#'
#' @return [`CallbackSet`]
#'
#' @export
callback_set = function(
  classname,
  # training
  on_begin = NULL,
  on_end = NULL,
  on_exit = NULL,
  on_epoch_begin = NULL,
  on_before_valid = NULL,
  on_epoch_end = NULL,
  on_batch_begin = NULL,
  on_batch_end = NULL,
  on_after_backward = NULL,
  # validation
  on_batch_valid_begin = NULL,
  on_batch_valid_end = NULL,
  on_valid_end = NULL,
  # other methods
  state_dict = NULL,
  load_state_dict = NULL,
  initialize = NULL,
  public = NULL, private = NULL, active = NULL, parent_env = parent.frame(), inherit = CallbackSet,
  lock_objects = FALSE
  ) {
  assert_true(startsWith(classname, "CallbackSet"))
  assert_false(xor(is.null(state_dict), is.null(load_state_dict)),
    .var.name = "Implement both state_dict and load_state_dict")
  assert_function(state_dict, nargs = 0, null.ok = TRUE)
  assert_function(load_state_dict, args = "state_dict", nargs = 1, null.ok = TRUE)
  more_public = list(
    on_begin = assert_function(on_begin, nargs = 0, null.ok = TRUE),
    on_end = assert_function(on_end, nargs = 0, null.ok = TRUE),
    on_epoch_begin = assert_function(on_epoch_begin, nargs = 0, null.ok = TRUE),
    on_before_valid = assert_function(on_before_valid, nargs = 0, null.ok = TRUE),
    on_epoch_end = assert_function(on_epoch_end, nargs = 0, null.ok = TRUE),
    on_batch_begin = assert_function(on_batch_begin, nargs = 0, null.ok = TRUE),
    on_batch_end = assert_function(on_batch_end, nargs = 0, null.ok = TRUE),
    on_after_backward = assert_function(on_after_backward, nargs = 0, null.ok = TRUE),
    on_batch_valid_begin = assert_function(on_batch_valid_begin, nargs = 0, null.ok = TRUE),
    on_batch_valid_end = assert_function(on_batch_valid_end, nargs = 0, null.ok = TRUE),
    on_valid_end = assert_function(on_valid_end, nargs = 0, null.ok = TRUE),
    on_exit = assert_function(on_exit, nargs = 0, null.ok = TRUE)
  )

  assert_function(initialize, null.ok = TRUE)

  if (!is.null(initialize)) {
    assert_true("initialize" %nin% names(public))
    public$initialize = initialize
  }

  assert_list(public, null.ok = TRUE, names = "unique")
  if (length(public)) assert_names(names(public), disjunct.from = names(more_public))

  if (!is.null(state_dict)) {
    public$state_dict = state_dict
    public$load_state_dict = load_state_dict
  }

  invalid_stages = names(public)[grepl("^on_", names(public))]

  if (length(invalid_stages)) {
    warningf("There are public method(s) with name(s) %s, which are not valid stages.",
      paste(paste0("'", invalid_stages, "'"), collapse = ", ")
    )
  }
  cloneable = test_function(private$deep_clone, args = c("name", "value"))

  assert_list(private, null.ok = TRUE, names = "unique")
  assert_list(active, null.ok = TRUE, names = "unique")
  assert_environment(parent_env)
  assert_inherits_classname(inherit, "CallbackSet")

  more_public = Filter(function(x) !is.null(x), more_public)
  parent_env_shim = new.env(parent = parent_env)
  parent_env_shim$inherit = inherit
  R6::R6Class(classname = classname, inherit = inherit, public = c(public, more_public),
    private = private, active = active, parent_env = parent_env_shim, lock_objects = lock_objects,
    cloneable = cloneable
  )
}
