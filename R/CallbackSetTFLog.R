#' @title TensorFlow Logging Callback
#'
#' @name mlr_callback_set.tflog
#'
#' @description
#' Logs the training and validation measures for tracking via TensorBoard.
#' @details
#' TODO: add
#'
#' @param path (`character(1)`)\cr
#'   The path to a folder where the events are logged. 
#'   Point TensorBoard to this folder to view them.
#' @family Callback
#' @export
#' @include CallbackSet.R
CallbackSetTFLog = R6Class("CallbackSetTFLog",
    inherit = CallbackSet,
    lock_objects = TRUE,
    public = list(
        #' @description
        #' Creates a new instance of this [R6][R6::R6Class] class.
        initialize = function(path = get_default_logdir()) {
            self$path = assert_path_for_output(path)
            set_default_logdir(path)
        },
        #' @description
        #' Logs the training measures as TensorFlow events.
        #' Meaningful changes happen at the end of each batch, 
        #' since this is when the gradient step occurs.
        on_batch_end = function() {
            log_train_score = function(measure_name) {
                train_score = list(self$ctx$last_scores_train[[measure_name]])
                names(train_score) = paste0("train.", measure_name)
                do.call(log_event, train_score)
            }

            if (length(self$ctx$last_scores_train)) {
                map(names(self$ctx$measures_train), log_train_score)
            }
        },
        #' @description
        #' Logs the validation measures as TensorFlow events.
        #' Meaningful changes happen at the end of each epoch.
        #' Notably NOT on_batch_valid_end, since there are no gradient steps between validation batches,
        #' and therefore differences are due to randomness
        on_epoch_end = function() {
            log_valid_score = function(measure_name) {
                valid_score = list(self$ctx$last_scores_valid[[measure_name]])
                names(valid_score) = paste0("valid.", measure_name)
                do.call(log_event, valid_score)
            }

            if (length(self$ctx$last_scores_valid)) {
                map(names(self$ctx$measure_valid), log_valid_score)
            }
        }
    )
)

mlr3torch_callbacks$add("tflog", function() {
    TorchCallback$new(
        callback_generator = CallbackSetCheckpoint,
        param_set = ps(
            path      = p_uty(tags = c("train", "required"))
        ),
        id = "tflog",
        label = "TFLog",
        man = "mlr3torch::mlr_callback_set.tflog"
    )
})
