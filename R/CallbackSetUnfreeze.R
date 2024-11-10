#' @title Unfreezing Weights Callback
#' 
#' @name mlr_callback_set.unfreeze
#' 
#' @description 
#' Freeze some weights for some number of steps or epochs.
#' 
#' @details 
#' TODO: add
#' 
#' @param starting_weights (`Selector`)\cr
#'  A `Selector` denoting the weights that are trainable from the start.
#' @param unfreeze (`data.table`)\cr
#'  A `data.table` with a column `weights` (a list column containing a `Selector`) and a column `epoch` or `batch`.
#' 
#' @family Callback
#' @export 
#' @include CallbackSet.R
CallbackSetFreeze = R6Class("CallbackSetUnfreeze",
  inherit = CallbackSet,
  lock_objects = FALSE,
  public = list(
    initialize = function() {

    }
  ),
  private = list(

  )
)

@include TorchCallback.R
mlr3torch_callbacks$add("unfreeze", function() {
  TorchCallback$new(
    callabck_generator = CallabckSetFreeze,
    param_set = ps(
      starting_weights = p_uty(tags = c("train", "required")),
      unfreeze = p_uty(tags = c("train", "required"))
    ),
    id = "unfreeze",
    label = "Unfreeze",
    man = "mlr3torch::mlr_callback_set.unfreeze"
  )
})