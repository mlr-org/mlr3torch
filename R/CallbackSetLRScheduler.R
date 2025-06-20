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
#' @param .scheduler (`lr_scheduler_generator`)\cr
#'   The `torch` scheduler generator (e.g. `torch::lr_step`).
#' @param ... (any)\cr
#'   The scheduler-specific initialization arguments.
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
    #' @param .scheduler (`LRScheduler`)\cr
    #' The learning rate scheduler wrapped by this callback.
    #' @param step_on_epoch (`logical(1)`)\cr
    #'   Whether the scheduler steps after every epoch (otherwise every batch).
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
      self$scheduler = invoke(self$scheduler_fn, optimizer = self$ctx$optimizer, .args = private$.scheduler_args)
    }
  ),
  private = list(
    .scheduler_args = NULL
  )
)

CallbackSetLRSchedulerOneCycle = R6Class("CallbackSetLRSchedulerOneCycle",
  inherit = CallbackSetLRScheduler,
  lock_objects = FALSE,
  public = list(
    on_begin = function() {
      private$.scheduler_args = insert_named(
        private$.scheduler_args,
        list(epochs = self$ctx$total_epochs, steps_per_epoch = self$ctx$loader_train$.length())
      )

      self$scheduler = invoke(self$scheduler_fn, optimizer = self$ctx$optimizer, .args = private$.scheduler_args)
    }
  )
)

CallbackSetLRSchedulerReduceOnPlateau = R6Class("CallbackSetLRSchedulerReduceOnPlateau",
  inherit = CallbackSetLRScheduler,
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
    #' @param step_on_epoch (`logical(1)`)\cr
    #'   Whether the scheduler steps after every epoch (otherwise every batch).
    initialize = function(.scheduler, step_on_epoch, ...) {
      assert_class(.scheduler, "lr_scheduler_generator")
      assert_flag(step_on_epoch)

      self$scheduler_fn = .scheduler
      private$.scheduler_args = list(...)

      self$on_epoch_end = function() {
        self$scheduler$step(self$ctx$last_scores_valid[[1L]])
      }
    },
    #' @description
    #' Creates the scheduler using the optimizer from the context
    on_begin = function() {
      if (class(self$scheduler_fn)[[1L]] == "lr_one_cycle") {
        private$.scheduler_args = insert_named(
          private$.scheduler_args,
          list(epochs = self$ctx$total_epochs, steps_per_epoch = self$ctx$loader_train$.length())
        )
      }

      self$scheduler = invoke(self$scheduler_fn, optimizer = self$ctx$optimizer, .args = private$.scheduler_args)
    }
  ),
  private = list(
    .scheduler_args = NULL
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
    man = "mlr3torch::mlr_callback_set.lr_scheduler",
    additional_args = list(.scheduler = torch::lr_one_cycle, step_on_epoch = FALSE)
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
    man = "mlr3torch::mlr_callback_set.lr_scheduler",
    additional_args = list(.scheduler = torch::lr_reduce_on_plateau, step_on_epoch = TRUE)
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
#' @param step_takes_valid_metric (`logical(1)`)\cr
#'   Whether the scheduler's `$step()` function takes a validation metric as an argument.
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