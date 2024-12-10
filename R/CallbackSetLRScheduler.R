#' @title Learning Rate Scheduling Callback
#' 
#' @name mlr_callback_set.lr_scheduler
#' 
#' @description 
#' Changes the learning rate based on the schedule specified by a `torch::LRScheduler`.
#' 
#' @param scheduler_fn (`function`)\cr
#'   The torch scheduler constructor function (e.g. `torch::lr_scheduler_step_lr`).
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
    #' @description
    #' Creates a new instance of this [R6][R6::R6Class] class.
    initialize = function(scheduler_fn, scheduler_step_on = "epoch") {
      self$scheduler_fn = scheduler_fn
      self$step_on = step_on
    },

    #' @description
    #' Creates the scheduler using the optimizer from the context
    on_begin = function() {
      self$scheduler = self$scheduler_fn(self$ctx$optimizer)
    },

    #' @description
    #' Depending on the scheduler, step after each epoch
    on_epoch_end = function() {
      if (self$step_on == "epoch") {
        self$scheduler$step()
      }
    },

    #' @description
    #' Depending on the scheduler, step after each batch
    on_batch_end = function() {
      if (self$step_on == "batch") {
        self$scheduler$step()
      }
    }
  )
)

#' @include TorchCallback.R
mlr3torch_callbacks$add("lr_scheduler", function() {
  TorchCallback$new(
    callback_generator = CallbackSetLRScheduler,
    param_set = ps(
      scheduler_fn = p_uty(tags = c("train", "required"), custom_check = function(input) check_class(input, "LRScheduler")),
      scheduler_args = p_uty(default = list(), tags = "train"),
      step_on = p_fct(levels = c("epoch", "batch"), default = "epoch", tags = "train")
    ),
    id = "lr_scheduler",
    label = "Learning Rate Scheduler",
    man = "mlr3torch::mlr_callback_set.lr_scheduler"
  )
})