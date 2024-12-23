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
#'   When the scheduler updates the learning rate. Must be:
#'   * "epoch" - Step after each epoch (default)
#'
#' 
#' @export
CallbackSetLRScheduler = R6Class("CallbackSetLRScheduler",
  inherit = CallbackSet,
  public = list(
    #' @description
    #' Creates a new instance of this [R6][R6::R6Class] class.
    initialize = function(.scheduler, ...) {
      
    },

    #' @description
    #' Creates the scheduler using the optimizer from the context
    on_begin = function() {
      self$scheduler = invoke(self$scheduler_fn, optimizer = self$ctx$optimizer)
    },

    #' @description
    #' Depending on the scheduler, step after each epoch
    on_epoch_end = function() {
      self$scheduler$step()
    }
    # TODO: add batches
)

# TODO: add custom schedulers (ignore for MVP)

#' @include TorchCallback.R
mlr3torch_callbacks$add("lr_scheduler_step", function() {
  TorchCallback$new(
    callback_generator = CallbackSetLRScheduler,
    param_set = ps(
      scheduler = p
      a = p_int()
    ),
    id = "lr_scheduler",
    label = "Learning Rate Scheduler",
    man = "mlr3torch::mlr_callback_set.lr_scheduler",
    additional_args = lr_step
  )
})

#' @include TorchCallback.R
mlr3torch_callbacks$add("lr_scheduler_cosine_annealing", function() {
  TorchCallback$new(
    callback_generator = CallbackSetLRScheduler,
    param_set = ps(
      # scheduler_fn_2 = p_uty(tags = c("train", "required"), custom_check = function(input) check_class(input, "LRScheduler")),
      # scheduler_args = p_uty(default = list(), tags = "train"),
      # step_on = p_fct(levels = c("epoch", "batch"), default = "epoch", tags = "train"),
      b = p_int()
    ),
    id = "lr_scheduler",
    label = "Learning Rate Scheduler",
    man = "mlr3torch::mlr_callback_set.lr_scheduler",
    additional_args = lr_cosine_annealing
  )
})

#' @include TorchCallback.R
mlr3torch_callbacks$add("lr_scheduler_custom", function() {
  TorchCallback$new(
    callback_generator = CallbackSetLRScheduler,
    param_set = ps(
      .scheduler = p_uty(tags = c("train", "required"), custom_check = function(input) check_class(input, "LRScheduler")),
      
    ),
    id = "lr_scheduler",
    label = "Learning Rate Scheduler",
    man = "mlr3torch::mlr_callback_set.lr_scheduler",
    # additional_args = lr_cosine_annealing
  )
})

as_lr_scheduler = function(lr_scheduler) {
  # infer the ps from the lr_scheduler signature (using inferps())

  # alternatively, allow the user to pass in a ps
}

t_clbk("lr_step", ...)

custom_scheduler = function()
as_lr_scheduler(custom_scheduler)