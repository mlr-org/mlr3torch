#' @title Unfreezing Weights Callback
#'
#' @name mlr_callback_set.unfreeze
#'
#' @description
#' Unfreeze some weights (parameters of the network) after some number of steps or epochs.
#'
#' @param starting_weights (`Select`)\cr
#'  A `Select` denoting the weights that are trainable from the start.
#' @param unfreeze (`data.table`)\cr
#'  A `data.table` with a column `weights` (a list column containing a `Select`) and a column `epoch` or `batch`.
#'
#' @family Callback
#' @export
#' @include CallbackSet.R
CallbackSetUnfreeze = R6Class("CallbackSetUnfreeze",
  inherit = CallbackSet,
  lock_objects = FALSE,
  public = list(
    #' @description
    #' Creates a new instance of this [R6][R6::R6Class] class.
    initialize = function(starting_weights, unfreeze) {
      self$starting_weights = starting_weights
      # consider supporting character vectors
      self$unfreeze = unfreeze
    },
    #' @description
    #' Sets the starting weights
    on_begin = function() {
      trainable_weights = self$starting_weights(names(self$ctx$network$parameters))
      walk(self$ctx$network$parameters[trainable_weights], function(param) param$requires_grad_(TRUE))
      frozen_weights = select_invert(self$starting_weights)(names(self$ctx$network$parameters))
      walk(self$ctx$network$parameters[frozen_weights], function(param) param$requires_grad_(FALSE))
    },
    #' @description
    #' Unfreezes weights if the training is at the correct epoch
    on_epoch_begin = function() {
      if ("epoch" %in% names(self$unfreeze)) {
        if (self$ctx$epoch %in% self$unfreeze$epoch) {
          weights = (self$unfreeze[epoch == self$ctx$epoch]$unfreeze)[[1]](names(self$ctx$network$parameters))
          walk(self$ctx$network$parameters[weights], function(param) param$requires_grad_(TRUE))
        }
      }
    },
    #' @description
    #' Unfreezes weights if the training is at the correct batch
    on_batch_begin = function() {
      if ("batch" %in% names(self$unfreeze)) {
        batch_num = (self$ctx$epoch - 1) * length(self$ctx$loader_train) + self$ctx$step
        if (batch_num %in% self$unfreeze$batch) {
          weights = (self$unfreeze[batch == batch_num]$unfreeze)[[1]](names(self$ctx$network$parameters))
          walk(self$ctx$network$parameters[weights], function(param) param$requires_grad_(TRUE))
        }
      }
    }
  )
)

#' @include TorchCallback.R
mlr3torch_callbacks$add("unfreeze", function() {
  TorchCallback$new(
    callback_generator = CallbackSetUnfreeze,
    param_set = ps(
      starting_weights = p_uty(
        tags = c("train", "required"),
        custom_check = function(input) {
          # check that the input is a selector
          # check that the selector is valid(?)
        }),
      unfreeze = p_uty(
        tags = c("train", "required"),
        custom_check = function(input) {
          # check that this is a data.table
          # check that one of the columns is named `epoch` or `batch`
          # check that there is only one column named `epoch` or `batch``
          # check that the selectors are valid(?)
        }
      )
    ),
    id = "unfreeze",
    label = "Unfreeze",
    man = "mlr3torch::mlr_callback_set.unfreeze"
  )
})
