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
#' @param starting_weights (`SelectorParam`)\cr
#'  A `Selector` denoting the weights that are trainable from the start.
#' @param unfreeze (`data.table`)\cr
#'  A `data.table` with a column `weights` (a list column containing a `SelectorParam`) and a column `epoch` or `batch`.
#' 
#' @family Callback
#' @export 
#' @include CallbackSet.R
CallbackSetUnfreeze = R6Class("CallbackSetUnfreeze",
  inherit = CallbackSet,
  lock_objects = FALSE,
  public = list(
    initialize = function(starting_weights, unfreeze) {
      self$starting_weights = starting_weights
      self$unfreeze = unfreeze

      self$batch_num = 0

      # sort the unfreeze data.table??
    
      weights = selectorparam_invert(starting_weights)
      mlr3misc::map(self$ctx$network$parameters[weights], function(param) param$requires_grad_(FALSE))
    },
    on_batch_end = function() {
      self$batch_num = self$batch_num + 1
    },
    on_epoch_begin = function() {
      if (self$ctx$epoch %in% self$unfreeze$epoch) {
        # get the specific layer
        weights = self$unfreeze[epoch == self$ctx$epoch]$weights
        mlr3misc::map(self$ctx$network$parameters[weights], function(param) param$requires_grad_(FALSE))
      }
    },
    on_batch_begin = function() {
      if (self$batch_num %in% self$unfreeze$batch) {
        weights = self$unfreeze[batch == self$batch_num]$weights
        mlr3misc::map(self$ctx$network$parameters[weights], function(param) param$requires_grad_(FALSE))
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