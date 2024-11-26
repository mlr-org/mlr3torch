#' @title Unfreezing Weights Callback
#'
#' @name mlr_callback_set.unfreeze
#'
#' @description
#' Unfreeze some weights after some number of steps or epochs. Select either a given module or a parameter.
#'
#' @details
#' TODO: add
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

      # TODO: remove, you can access this or compute from the information in the context
      self$batch_num = 0

      # sort the unfreeze data.table??
    },
    #' @description
    #' Sets the starting weights
    on_begin = function() {
      weights = select_invert(self$starting_weights)(names(self$ctx$network$parameters))
      walk(self$ctx$network$parameters[weights], function(param) param$requires_grad_(FALSE))
    },
    #' @description
    #' Increment the batch counter (old)
    on_batch_end = function() {
      # TODO: compute from epoch and step (batch num within an epoch)
      self$batch_num = self$batch_num + 1
    },
    #' @description
    #' Unfreezes weights if the training is at the correct epoch
    on_epoch_begin = function() {
      if (self$ctx$epoch %in% self$unfreeze$epoch) {
        # debugonce()
        weights = (self$unfreeze[epoch == self$ctx$epoch]$unfreeze)[[1]](names(self$ctx$network$parameters))
        browser()
        walk(self$ctx$network$parameters[weights], function(param) param$requires_grad_(TRUE))
      }
    },
    #' @description
    #' Unfreezes weights if the training is at the correct batch
    on_batch_begin = function() {
      if (self$batch_num %in% self$unfreeze$batch) {
        weights = (self$unfreeze[epoch == self$ctx$epoch]$unfreeze)[[1]](names(self$ctx$network$parameters))
        walk(self$ctx$network$parameters[weights], function(param) param$requires_grad_(TRUE))
      }
    }
  ),
  private = list()
)

#' @include TorchCallback.R
mlr3torch_callbacks$add("unfreeze", function() {
  TorchCallback$new(
    callback_generator = CallbackSetUnfreeze,
    param_set = ps(
      starting_weights = p_uty(tags = c("train", "required")),
      unfreeze = p_uty(tags = c("train", "required"))
    ),
    id = "unfreeze",
    label = "Unfreeze",
    man = "mlr3torch::mlr_callback_set.unfreeze"
  )
})
