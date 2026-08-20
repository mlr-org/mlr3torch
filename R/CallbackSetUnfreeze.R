#' @title Unfreezing Weights Callback
#'
#' @name mlr_callback_set.unfreeze
#'
#' @description
#' Unfreeze some weights (parameters of the network) after some number of steps or epochs.
#'
#' @section Resuming:
#' Which weights are trainable is stored and restored, so a resumed run does not freeze again what
#' the run it continues had already unfrozen: `starting_weights` is applied first, and the restored
#' weights are unfrozen on top of it.
#'
#' @param starting_weights (`Select`)\cr
#'  A `Select` denoting the weights that are trainable from the start.
#' @param unfreeze (`data.table`)\cr
#'  A `data.table` with a column `weights` (a list column of `Select`s) and a column `epoch` or `batch`.
#'  The selector indicates which parameters to unfreeze, while the `epoch` or `batch` column indicates when to do so.
#'
#' @family Callback
#' @export
#' @include CallbackSet.R
#' @examplesIf torch::torch_is_installed()
#' task = tsk("iris")
#' cb = t_clbk("unfreeze")
#' mlp = lrn("classif.mlp", callbacks = cb,
#'  cb.unfreeze.starting_weights = select_invert(
#'    select_name(c("0.weight", "3.weight", "6.weight", "6.bias"))
#'  ),
#'  cb.unfreeze.unfreeze = data.table(
#'    epoch = c(2, 5),
#'    weights = list(select_name("0.weight"), select_name(c("3.weight", "6.weight")))
#'  ),
#'  epochs = 6, batch_size = 150, neurons = c(1, 1, 1)
#' )
#'
#' mlp$train(task)
CallbackSetUnfreeze = R6Class("CallbackSetUnfreeze",
  inherit = CallbackSet,
  lock_objects = FALSE,
  public = list(
    #' @description
    #' Creates a new instance of this [R6][R6::R6Class] class.
    initialize = function(starting_weights, unfreeze) {
      self$starting_weights = starting_weights
      self$unfreeze = unfreeze
      private$.batchwise = "batch" %in% names(self$unfreeze)
    },
    #' @description
    #' Sets the starting weights
    on_begin = function() {
      trainable_weights = self$starting_weights(names(self$ctx$network$parameters))
      walk(self$ctx$network$parameters[trainable_weights], function(param) param$requires_grad_(TRUE))
      frozen_weights = select_invert(self$starting_weights)(names(self$ctx$network$parameters))
      walk(self$ctx$network$parameters[frozen_weights], function(param) param$requires_grad_(FALSE))

      frozen_weights_str = paste(trainable_weights, collapse = ", ")
      lg$info(sprintf("Training the following weights at the start: %s", paste0(trainable_weights, collapse = ", ")))

      private$.restore_trainable()
    },
    #' @description
    #' Returns the names of the weights that are currently trainable, so that a later run does not
    #' freeze weights again that were already unfrozen.
    state_dict = function() {
      list(trainable = private$.trainable_weights())
    },
    #' @description
    #' Marks the weights of the state dict as trainable.
    #' @param state_dict (named `list()`)\cr
    #'   The state dict as retrieved via `$state_dict()`.
    load_state_dict = function(state_dict) {
      # the network is frozen according to `starting_weights` in $on_begin(), so the weights that
      # were already unfrozen can only be restored afterwards
      private$.prev_state = state_dict
      invisible(NULL)
    },
    #' @description
    #' Unfreezes weights if the training is at the correct epoch
    on_epoch_begin = function() {
      if (!private$.batchwise) {
        if (self$ctx$epoch %in% self$unfreeze$epoch) {
          weights = (self$unfreeze[get("epoch") == self$ctx$epoch]$weights)[[1]](names(self$ctx$network$parameters))
          if (!length(weights)) {
            lg$warn(paste0("No weights unfrozen at epoch ", self$ctx$epoch, " , check the specification of the Selector"))
          } else {
            walk(self$ctx$network$parameters[weights], function(param) param$requires_grad_(TRUE))
            weights_str = paste(weights, collapse = ", ")
            lg$info(paste0("Unfreezing at epoch ", self$ctx$epoch, ": ", weights_str))
          }

        }
      }
    },
    #' @description
    #' Unfreezes weights if the training is at the correct batch
    on_batch_begin = function() {
      if (private$.batchwise) {
        batch_num = self$ctx$global_step
        if (batch_num %in% self$unfreeze$batch) {
          weights = (self$unfreeze[get("batch") == batch_num]$weights)[[1]](names(self$ctx$network$parameters))
          if (!length(weights)) {
            lg$warn(paste0("No weights unfrozen at batch ", batch_num, " , check the specification of the Selector"))
          } else {
            walk(self$ctx$network$parameters[weights], function(param) param$requires_grad_(TRUE))
            weights_str = paste(weights, collapse = ", ")
            lg$info(paste0("Unfreezing at batch ", batch_num, ": ", weights_str))
          }
        }
      }
    }
  ),
  private = list(
    .batchwise = NULL,
    .prev_state = NULL,
    .trainable_weights = function() {
      params = self$ctx$network$parameters
      names(params)[map_lgl(params, function(param) param$requires_grad)]
    },
    .restore_trainable = function() {
      if (is.null(private$.prev_state)) return(NULL)
      weights = intersect(private$.prev_state$trainable, names(self$ctx$network$parameters))
      walk(self$ctx$network$parameters[weights], function(param) param$requires_grad_(TRUE))
      lg$info(sprintf("Restoring the following trainable weights: %s", paste0(weights, collapse = ", ")))
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
        custom_check = function(input) check_class(input, "Select")
      ),
      unfreeze = p_uty(
        tags = c("train", "required"),
        custom_check = check_unfreeze_dt
      )
    ),
    id = "unfreeze",
    label = "Unfreeze",
    man = "mlr3torch::mlr_callback_set.unfreeze"
  )
})

check_unfreeze_dt = function(x) {
  if (is.null(x) || (is.data.table(x) && nrow(x) == 0)) {
    return(TRUE)
  }
  if (!test_class(x, "data.table")) {
    return("`unfreeze` must be a data.table()")
  }
  if (!test_names(names(x), must.include = "weights")) {
    return("Must contain 2 columns: `weights` and (epoch or batch)")
  }
  if (!xor("epoch" %in% names(x), "batch" %in% names(x))) {
    return("Exactly one of the columns must be named 'epoch' or 'batch'")
  }
  xs = x[["epoch"]] %??% x[["batch"]]
  if (!test_integerish(xs, lower = 0L) || anyDuplicated(xs)) {
    return("Column batch/epoch must be a positive integerish vector without duplicates.")
  }
  if (!test_list(x$weights)) {
    return("The `weights` column should be a list")
  }
  if (some(x$weights, function(input) !test_class(input, classes = "Select"))) {
    return("The `weights` column should be a list of Selects")
  }
  return(TRUE)
}
