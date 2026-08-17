#' @title Learning Rate Scheduling Callback
#'
#' @name mlr_callback_set.lr_scheduler
#'
#' @description
#' Changes the learning rate based on the schedule specified by a `torch::lr_scheduler`.
#'
#' As of this writing, the following are available:
#'
#' * [torch::lr_cosine_annealing()]
#' * [torch::lr_lambda()]
#' * [torch::lr_multiplicative()]
#' * [torch::lr_one_cycle()] (where the default values for `epochs` and `steps_per_epoch` are the number of training epochs and the number of batches per epoch)
#' * [torch::lr_reduce_on_plateau()]
#' * [torch::lr_step()]
#' * Custom schedulers defined with [torch::lr_scheduler()].
#'
#' Attaching more than one of these is possible but rarely what you want, as they are not
#' independent: a schedule that sets the rate from the initial one ([torch::lr_lambda()]) overwrites
#' what a schedule that scales the current one ([torch::lr_step()], [torch::lr_multiplicative()])
#' did in the same epoch, so which one has an effect comes down to the order the callbacks were
#' passed in (see the section *Ordering* of [`CallbackSet`]).
#'
#' @section Resuming:
#' The state of the wrapped `torch` scheduler is stored and restored, so a resumed run continues the
#' schedule instead of starting it over.
#' Creating a scheduler resets the optimizer's learning rate to the one the schedule started at, so
#' the rate the restored schedule had reached is put back afterwards.
#'
#' That state contains the scheduler's configuration as well as its progress, and restoring it
#' overwrites what the resuming run was configured with.
#' Resuming with different scheduler arguments -- or a different `opt.lr`, which the schedule's base
#' rates are derived from -- therefore silently continues the schedule of the checkpointed run.
#' Configure both runs the same way; a schedule cannot be changed halfway through.
#'
#' @param .scheduler (`lr_scheduler_generator`)\cr
#'   The `torch` scheduler generator (e.g. `torch::lr_step`).
#' @param ... (any)\cr
#'   The scheduler-specific initialization arguments.
#' @param step_on_epoch (`logical(1)`)\cr
#'   Whether the scheduler steps after every epoch (otherwise every batch).
#'
#' @export
CallbackSetLRScheduler = R6Class("CallbackSetLRScheduler",
  inherit = CallbackSet,
  lock_objects = FALSE,
  public = list(
    #' @field scheduler_fn (`lr_scheduler_generator`)\cr
    #' The `torch` function that creates a learning rate scheduler
    scheduler_fn = NULL,
    #' @field scheduler (`LRScheduler`)\cr
    #' The learning rate scheduler wrapped by this callback
    scheduler = NULL,
    #' @description
    #' Creates a new instance of this [R6][R6::R6Class] class.
    initialize = function(.scheduler, step_on_epoch, ...) {
      assert_class(.scheduler, "lr_scheduler_generator")
      assert_flag(step_on_epoch)

      self$scheduler_fn = .scheduler
      private$.scheduler_args = list(...)

      if (step_on_epoch) {
        self$on_epoch_end = function() self$scheduler$step()
      } else {
        self$on_batch_end = function() self$scheduler$step()
      }
    },
    #' @description
    #' Creates the scheduler using the optimizer from the context
    on_begin = function() {
      # Creating a scheduler sets the optimizer's learning rate back to the `initial_lr` that the
      # first scheduler recorded on it. When the optimizer was restored from a checkpoint that is
      # the rate the schedule *started* at, not the one it had reached, so it is remembered here
      # and put back in $.restore_scheduler_state().
      lrs = map_dbl(self$ctx$optimizer$param_groups, function(group) group$lr)
      self$scheduler = invoke(self$scheduler_fn, optimizer = self$ctx$optimizer, .args = private$.scheduler_args)
      private$.restore_scheduler_state(lrs)
    },
    #' @description
    #' Returns the state of the wrapped `torch` scheduler, so that a later run can continue the
    #' schedule instead of starting it over.
    #' Returns `NULL` if the scheduler was not created yet, i.e. before the training loop began.
    state_dict = function() {
      if (!is.null(self$scheduler)) self$scheduler$state_dict()
    },
    #' @description
    #' Loads the state of the wrapped `torch` scheduler.
    #' @param state_dict (named `list()`)\cr
    #'   The state dict as retrieved via `$state_dict()`.
    load_state_dict = function(state_dict) {
      # the scheduler only exists once the training loop has begun, so the state is applied in
      # $on_begin() and merely remembered here
      private$.prev_state = state_dict
      if (!is.null(self$scheduler)) private$.restore_scheduler_state()
      invisible(NULL)
    }
  ),
  private = list(
    .scheduler_args = NULL,
    .prev_state = NULL,
    # `lrs` are the learning rates the optimizer had before the scheduler was created, or NULL when
    # the state is restored while the scheduler already exists.
    .restore_scheduler_state = function(lrs = NULL) {
      if (is.null(private$.prev_state)) return(NULL)
      # the state dict of the freshly created scheduler describes the schedule this run is
      # configured for, which is not necessarily the one the state was saved for
      private$.assert_compatible_state(self$scheduler$state_dict(), private$.prev_state)
      self$scheduler$load_state_dict(private$.prev_state)
      # `$load_state_dict()` only restores the scheduler, so the optimizer is still at the
      # `initial_lr` the constructor put it back to. Most schedules recompute the rate from
      # `base_lrs` and `last_epoch` on their next step and so repair themselves, but one that
      # computes it from the current rate -- `lr_multiplicative` -- would start over from there.
      if (!is.null(lrs)) {
        walk(seq_along(self$ctx$optimizer$param_groups), function(i) {
          self$ctx$optimizer$param_groups[[i]]$lr = lrs[[i]]
        })
      }
      private$.prev_state = NULL
    },
    # Schedulers whose shape depends on the length of the training run overwrite this to fail
    # before the first epoch instead of somewhere in the middle of the run.
    .assert_compatible_state = function(current, restored) NULL
  )
)

#' @title OneCycle Learning Rate Scheduling Callback
#'
#' @name mlr_callback_set.lr_scheduler_one_cycle
#'
#' @description
#' Changes the learning rate based on the 1cycle learning rate policy.
#'
#' Wraps [torch::lr_one_cycle()], where the default values for `epochs` and `steps_per_epoch` are the number of training epochs and the number of batches per epoch.
#'
#' @section Resuming:
#' As for [`CallbackSetLRScheduler`], with one additional restriction: the 1cycle policy is defined
#' over the total number of steps of the run, so a resumed run must be configured for the same
#' number of steps as the one that wrote the checkpoint, i.e. the same `epochs` and the same number
#' of batches per epoch.
#' A run that is not errors before its first epoch rather than somewhere in the middle.
#'
#' @param ... (any)\cr
#'   The scheduler-specific initialization arguments.
#'
#' @export
CallbackSetLRSchedulerOneCycle = R6Class("CallbackSetLRSchedulerOneCycle",
  inherit = CallbackSetLRScheduler,
  lock_objects = FALSE,
  public = list(
    #' @description
    #' Creates a new instance of this [R6][R6::R6Class] class.
    initialize = function(...) {
      super$initialize(
        .scheduler = torch::lr_one_cycle,
        step_on_epoch = FALSE,
        ...
        )
    },
    #' @description
    #' Creates the scheduler using the optimizer from the context
    on_begin = function() {
      private$.scheduler_args = insert_named(
        private$.scheduler_args,
        list(epochs = self$ctx$total_epochs, steps_per_epoch = self$ctx$loader_train$.length())
      )

      super$on_begin()
    }
  ),
  private = list(
    # The one cycle policy is defined over the total number of steps, so a restored state is only
    # meaningful when this run has the same length as the one the state was saved in.
    # Without this check, torch errors in the middle of the run ("Tried to step ... times") or the
    # cycle silently never finishes annealing.
    .assert_compatible_state = function(current, restored) {
      if (isTRUE(all.equal(restored$total_steps, current$total_steps))) {
        return(NULL)
      }
      stopf("Cannot load the state of the one cycle learning rate schedule: it was saved for a schedule of %s total steps, but this run is configured for %s (epochs = %s). Both runs must be configured with the same 'epochs' and yield the same number of batches per epoch.", # nolint
        format(restored$total_steps %??% NA), format(current$total_steps), self$ctx$total_epochs)
    }
  )
)

#' @title Reduce On Plateau Learning Rate Scheduler
#'
#' @name mlr_callback_set.lr_scheduler_reduce_on_plateau
#'
#' @description
#' Reduces the learning rate when the first validation metric stops improving for `patience` epochs.
#' Wraps [torch::lr_reduce_on_plateau()]
#'
#' @section Resuming:
#' As for [`CallbackSetLRScheduler`]. For this schedule the restored state includes the best score
#' seen so far and how long it has been stagnating, so `patience` keeps counting across runs instead
#' of starting over.
#'
#' @param ... (any)\cr
#'   The scheduler-specific initialization arguments.
#'
#' @export
CallbackSetLRSchedulerReduceOnPlateau = R6Class("CallbackSetLRSchedulerReduceOnPlateau",
  inherit = CallbackSetLRScheduler,
  lock_objects = FALSE,
  public = list(
    #' @description
    #' Creates a new instance of this [R6][R6::R6Class] class.
    initialize = function(...) {
      super$initialize(
        .scheduler = torch::lr_reduce_on_plateau,
        step_on_epoch = TRUE,
        ...
      )

      self$on_epoch_end = function() {
        # `last_scores_valid` is NULL in epochs where no validation is performed
        # (i.e. when `eval_freq > 1`) and for the whole run when no validation is configured.
        scores = self$ctx$last_scores_valid
        if (!length(scores)) {
          return(NULL)
        }
        self$scheduler$step(scores[[1L]])
      }
    }
  )
)

# some of the schedulers accept lists
# so they can treat different parameter groups differently
check_class_or_list = function(x, classname) {
  if (is.list(x)) check_list(x, types = classname) else check_class(x, classname)
}

#' @include TorchCallback.R
mlr3torch_callbacks$add("lr_cosine_annealing", function() {
  TorchCallback$new(
    callback_generator = CallbackSetLRScheduler,
    param_set = ps(
      T_max = p_int(tags = c("train", "required")),
      eta_min = p_dbl(default = 0, tags = "train"),
      last_epoch = p_int(default = -1, tags = "train"),
      verbose = p_lgl(default = FALSE, tags = "train")
    ),
    id = "lr_cosine_annealing",
    label = "Cosine Annealing LR Scheduler",
    man = "mlr3torch::mlr_callback_set.lr_scheduler",
    additional_args = list(.scheduler = torch::lr_cosine_annealing, step_on_epoch = TRUE)
  )
})

#' @include TorchCallback.R
mlr3torch_callbacks$add("lr_lambda", function() {
  TorchCallback$new(
    callback_generator = CallbackSetLRScheduler,
    param_set = ps(
      lr_lambda = p_uty(tags = c("train", "required"), custom_check = function(x) check_class_or_list(x, "function")),
      last_epoch = p_int(default = -1, tags = "train"),
      verbose = p_lgl(default = FALSE, tags = "train")
    ),
    id = "lr_lambda",
    label = "Multiplication by Function LR Scheduler",
    man = "mlr3torch::mlr_callback_set.lr_scheduler",
    additional_args = list(.scheduler = torch::lr_lambda, step_on_epoch = TRUE)
  )
})

#' @include TorchCallback.R
mlr3torch_callbacks$add("lr_multiplicative", function() {
  TorchCallback$new(
    callback_generator = CallbackSetLRScheduler,
    param_set = ps(
      lr_lambda = p_uty(tags = c("train", "required"), custom_check = function(x) check_class_or_list(x, "function")),
      last_epoch = p_int(default = -1, tags = "train"),
      verbose = p_lgl(default = FALSE, tags = "train")
    ),
    id = "lr_multiplicative",
    label = "Multiplication by Factor LR Scheduler",
    man = "mlr3torch::mlr_callback_set.lr_scheduler",
    additional_args = list(.scheduler = torch::lr_multiplicative, step_on_epoch = TRUE)
  )
})

#' @include TorchCallback.R
mlr3torch_callbacks$add("lr_one_cycle", function() {
  TorchCallback$new(
    callback_generator = CallbackSetLRSchedulerOneCycle,
    param_set = ps(
      max_lr = p_uty(tags = c("train", "required"), custom_check = function(x) check_class_or_list(x, "numeric")),
      total_steps = p_int(default = NULL, special_vals = list(NULL), tags = "train"),
      epochs = p_int(default = NULL, special_vals = list(NULL), tags = "train"),
      steps_per_epoch = p_int(default = NULL, special_vals = list(NULL), tags = "train"),
      pct_start = p_dbl(default = 0.3, tags = "train"),
      anneal_strategy = p_fct(default = "cos", levels = c("cos", "linear")), # this is a string in the torch fn
      cycle_momentum = p_lgl(default = TRUE, tags = "train"),
      base_momentum = p_uty(default = 0.85, tags = "train", custom_check = function(x) check_class_or_list(x, "numeric")),
      max_momentum = p_uty(default = 0.95, tags = "train", custom_check = function(x) check_class_or_list(x, "numeric")),
      div_factor = p_dbl(default = 25, tags = "train"),
      final_div_factor = p_dbl(default = 1e4, tags = "train"),
      verbose = p_lgl(default = FALSE, tags = "train")
    ),
    id = "lr_one_cycle",
    label = "1cycle LR Scheduler",
    man = "mlr3torch::mlr_callback_set.lr_scheduler_one_cycle"
  )
})

#' @include TorchCallback.R
mlr3torch_callbacks$add("lr_reduce_on_plateau", function() {
  TorchCallback$new(
    callback_generator = CallbackSetLRSchedulerReduceOnPlateau,
    param_set = ps(
      mode = p_fct(default = "min", levels = c("min", "max"), tags = "train"),
      factor = p_dbl(default = 0.1, tags = "train"),
      patience = p_int(default = 10, tags = "train"),
      threshold = p_dbl(default = 1e-04, tags = "train"),
      threshold_mode = p_fct(default = "rel", levels = c("rel", "abs"), tags = "train"),
      cooldown = p_int(default = 0, tags = "train"),
      min_lr = p_uty(default = 0, tags = "train", custom_check = function(x) check_class_or_list(x, "numeric")),
      eps = p_dbl(default = 1e-08, tags = "train"),
      verbose = p_lgl(default = FALSE, tags = "train")
    ),
    id = "lr_reduce_on_plateau",
    label = "Reduce on Plateau LR Scheduler",
    man = "mlr3torch::mlr_callback_set.lr_scheduler_reduce_on_plateau"
  )
})

#' @include TorchCallback.R
mlr3torch_callbacks$add("lr_step", function() {
  TorchCallback$new(
    callback_generator = CallbackSetLRScheduler,
    param_set = ps(
      step_size = p_int(tags = c("train", "required")),
      gamma = p_dbl(default = 0.1, tags = "train"),
      last_epoch = p_int(default = -1, tags = "train")
    ),
    id = "lr_step",
    label = "Step Decay LR Scheduler",
    man = "mlr3torch::mlr_callback_set.lr_scheduler",
    additional_args = list(.scheduler = torch::lr_step, step_on_epoch = TRUE)
  )
})

#' @title Convert to CallbackSetLRScheduler
#'
#' @description
#' Convert a `torch` scheduler generator to a `CallbackSetLRScheduler`.
#'
#' @param x (`function`)\cr
#'   The `torch` scheduler generator defined using `torch::lr_scheduler()`.
#' @param step_on_epoch (`logical(1)`)\cr
#'   Whether the scheduler steps after every epoch
#' @export
as_lr_scheduler = function(x, step_on_epoch) {
  assert_class(x, "lr_scheduler_generator")
  assert_flag(step_on_epoch)

  class_name = class(x)[1L]

  TorchCallback$new(
    callback_generator = CallbackSetLRScheduler,
    param_set = inferps(x),
    id = if (class_name == "") "lr_custom" else class_name,
    label = "Custom LR Scheduler",
    man = "mlr3torch::mlr_callback_set.lr_scheduler",
    additional_args = list(.scheduler = x, step_on_epoch = step_on_epoch)
  )
}
