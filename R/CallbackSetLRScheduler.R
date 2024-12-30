#' @title Learning Rate Scheduling Callback
#'
#' @name mlr_callback_set.lr_scheduler
#'
#' @description
#' Changes the learning rate based on the schedule specified by a `torch::LRScheduler`.
#'
#' @param scheduler_fn (`function`)\cr
#'   The torch scheduler constructor function (e.g. `torch::lr_scheduler_step_lr`).
#' @param scheduler_args (`list`)\cr
#'   A named list specifying the additional arguments
#' @param step_on (`character(1)`)\cr
#'   When the scheduler updates the learning rate. Must be one of:
#'   * "epoch" - Step after each epoch (default)
#'   * "batch" - Step after each batch
#'
#'
#' @export
CallbackSetLRScheduler = R6Class("CallbackSetLRScheduler",
  inherit = CallbackSet,
  public = list(
    scheduler_fn = NULL,
    scheduler = NULL,
    #' @description
    #' Creates a new instance of this [R6][R6::R6Class] class.
    initialize = function(.scheduler, ...) {
      self$scheduler_fn = .scheduler
      private$.scheduler_args = list(...)
    },
    #' @description
    #' Creates the scheduler using the optimizer from the context
    on_begin = function() {
      # args = self$ctx$param_set$get_values(prefix = "lr_scheduler")
      
      # Combine with optimizer
      # args = c(list(optimizer = self$ctx$optimizer), args)
      self$scheduler = invoke(self$scheduler_fn, optimizer = self$ctx$optimizer, .args = private$.scheduler_args)
    },
    #' @description
    #' Depending on the scheduler, step after each epoch
    on_epoch_end = function() {
      # TODO: ensure that this happens after optimizer$step()
      # https://blogs.rstudio.com/ai/posts/2020-10-19-torch-image-classification/#training
      # but for now let's hope that it does
      self$scheduler$step()
    }
    # TODO: add batches
  ),
  private = list(
    .scheduler_args = NULL
  )
)

# TODO: determine whether we should set ranges and such even when torch does not

# some of the schedulers accept lists
# so they can treat different parameter groups differently
check_class_or_list = function(x, classname) {
  # TODO: make this work only for lists
  # currently vectors are allowed as well
  if (some(!map_lgl(x, test_class(x, classname)))) {
    return(paste0("One of the arguments is not of class ", classname))
  }
  return(TRUE)
}

# begin built-in torch LR schedulers

#' @include TorchCallback.R
mlr3torch_callbacks$add("lr_scheduler_cosine_annealing", function() {
  TorchCallback$new(
    callback_generator = CallbackSetLRScheduler,
    param_set = ps(
      T_max = p_int(tags = c("train", "required")),
      eta_min = p_dbl(default = 0, lower = 0, tags = "train"),
      last_epoch = p_int(default = -1, tags = "train"),
      verbose = p_lgl(default = FALSE, tags = "train")
    ),
    id = "lr_scheduler",
    label = "Learning Rate Scheduler",
    man = "mlr3torch::mlr_callback_set.lr_scheduler",
    additional_args = torch::lr_cosine_annealing
  )
})

#' @include TorchCallback.R
mlr3torch_callbacks$add("lr_scheduler_lambda", function() {
  TorchCallback$new(
    callback_generator = CallbackSetLRScheduler,
    param_set = ps(
      lr_lambda = p_uty(tags = c("train"), custom_check = function(x) check_class_or_list(x, "function")), # TODO: assert fn or list of fns
      last_epoch = p_int(default = -1, lower = -1, tags = "train"),
      verbose = p_lgl(default = FALSE, tags = "train")
    ),
    id = "lr_scheduler",
    label = "Learning Rate Scheduler",
    man = "mlr3torch::mlr_callback_set.lr_scheduler",
    additional_args = lr_lambda
  )
})

#' @include TorchCallback.R
mlr3torch_callbacks$add("lr_scheduler_multiplicative", function() {
  TorchCallback$new(
    callback_generator = CallbackSetLRScheduler,
    param_set = ps(
      lr_lambda = p_uty(tags = c("train"), custom_check = function(x) check_class_or_list(x, "function")),
      last_epoch = p_int(default = -1, lower = -1, tags = "train"),
      verbose = p_lgl(default = FALSE, tags = "train")
    ),
    id = "lr_scheduler",
    label = "Learning Rate Scheduler",
    man = "mlr3torch::mlr_callback_set.lr_scheduler",
    additional_args = lr_multiplicative
  )
})

#' @include TorchCallback.R
mlr3torch_callbacks$add("lr_scheduler_one_cycle", function() {
  TorchCallback$new(
    callback_generator = CallbackSetLRScheduler,
    param_set = ps(
      max_lr = p_dbl(tags = "train"),
      total_steps = p_int(default = NULL, tags = "train"),
      epochs = p_int(default = NULL, tags = "train"),
      steps_per_epoch = NULL,
      pct_start = p_dbl(default = 0.3, lower = 0, upper = 1, tags = "train"),
      anneal_strategy = p_fct(default = "cos", levels = c("cos", "linear")), # this is a string in the torch fn
      cycle_momentum = p_lgl(default = TRUE, tags = "train"),
      base_momentum = p_uty(default = 0.85, tags = "train", custom_check = function(x) check_class_or_list(x, "numeric")), # float or list
      max_momentum = p_uty(default = 0.95, tags = "train", custom_check = function(x) check_class_or_list(x, "numeric")), # or list
      div_factor = p_dbl(default = 25, tags = "train"),
      final_div_factor = p_dbl(default = 1e4, tags = "train"),
      verbose = p_lgl(default = FALSE, tags = "train")
    ),
    id = "lr_scheduler",
    label = "Learning Rate Scheduler",
    man = "mlr3torch::mlr_callback_set.lr_scheduler",
    additional_args = lr_one_cycle
  )
})

#' @include TorchCallback.R
mlr3torch_callbacks$add("lr_scheduler_reduce_on_plateau", function() {
  TorchCallback$new(
    callback_generator = CallbackSetLRScheduler,
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
    id = "lr_scheduler",
    label = "Learning Rate Scheduler",
    man = "mlr3torch::mlr_callback_set.lr_scheduler",
    additional_args = lr_reduce_on_plateau
  )
})

#' @include TorchCallback.R
mlr3torch_callbacks$add("lr_scheduler_step", function() {
  TorchCallback$new(
    callback_generator = CallbackSetLRScheduler,
    param_set = ps(
      step_size = p_int(default = 1, lower = 1, tags = "train"),
      gamma = p_dbl(default = 0.1, lower = 0, upper = 1, tags = "train"),
      last_epoch = p_int(default = -1, tags = "train")
    ),
    id = "lr_scheduler",
    label = "Learning Rate Scheduler",
    man = "mlr3torch::mlr_callback_set.lr_scheduler",
    additional_args = lr_step
  )
})

# end torch-provided schedulers

# #' @include TorchCallback.R
# mlr3torch_callbacks$add("lr_scheduler_custom", function() {
#   TorchCallback$new(
#     callback_generator = CallbackSetLRScheduler,
#     param_set = ps(
#       .scheduler = p_uty(tags = c("train", "required"), custom_check = function(input) check_class(input, "LRScheduler")),
#     ),
#     id = "lr_scheduler",
#     label = "Learning Rate Scheduler",
#     man = "mlr3torch::mlr_callback_set.lr_scheduler",
#     # additional_args = lr_cosine_annealing
#   )
# })

# # TODO: implement custom schedulers
# as_lr_scheduler = function(lr_scheduler) {
#   # infer the ps from the lr_scheduler signature (using inferps())

#   # alternatively, allow the user to pass in a ps
# }

# t_clbk("lr_step", ...)

# custom_scheduler = function()
# as_lr_scheduler(custom_scheduler)
