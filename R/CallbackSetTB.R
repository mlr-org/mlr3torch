#' @title TensorBoard Logging Callback
#'
#' @name mlr_callback_set.tb
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
CallbackSetTB = R6Class("CallbackSetTB",
    inherit = CallbackSet,
    lock_objects = TRUE,
    public = list(
        #' @description
        #' Creates a new instance of this [R6][R6::R6Class] class.
        initialize = function(path = tempfile()) {
            self$path = assert_path_for_output(path)
        },
        # #' @description
        # #' Logs the training measures as TensorFlow events.
        # #' Meaningful changes happen at the end of each batch, 
        # #' since this is when the gradient step occurs.
        # # TODO: change this to log last_loss
        # on_batch_end = function() {
        #     # TODO: determine whether you can refactor this and the 
        #     # validation one into a single function
        #     # need to be able to access self$ctx

        #     # TODO: pass in the appropriate step from the context
        #     log_event(last_loss = self$ctx$last_loss)
        # },
        #' @description
        #' Logs the validation measures as TensorFlow events.
        #' Meaningful changes happen at the end of each epoch.
        #' Notably NOT on_batch_valid_end, since there are no gradient steps between validation batches,
        #' and therefore differences are due to randomness
        # TODO: log last_scores_train here
        # TODO: display the appropriate x axis with its label in TensorBoard
        # relevant when we log different scores at different times
        on_epoch_end = function() {
            log_valid_score = function(measure_name) {
                valid_score = list(self$ctx$last_scores_valid[[measure_name]])
                names(valid_score) = paste0("valid.", measure_name)
                with_logdir(temp, {
                    do.call(log_event, valid_score)
                })
            }

            log_train_score = function(measure_name) {
                # TODO: change this to use last_loss. I don't recall why we wanted to do that.
                train_score = list(self$ctx$last_scores_train[[measure_name]])
                names(train_score) = paste0("train.", measure_name)
                with_logdir(temp, {
                    do.call(log_event, valid_score)
                })
            }

            if (length(self$ctx$last_scores_train)) {
                # TODO: decide whether we should put the temporary logdir modification here instead.
                map(names(self$ctx$measures_train), log_train_score)
            }

            if (length(self$ctx$last_scores_valid)) {
                map(names(self$ctx$measure_valid), log_valid_score)
            }
        }
    )
    # private = list(
    #     log_score = function(prefix, measure_name, score) {
            
    #     }
    # )
)


mlr3torch_callbacks$add("tb", function() {
    TorchCallback$new(
        callback_generator = CallbackSetCheckpoint,
        param_set = ps(
            path      = p_uty(tags = c("train", "required"))
        ),
        id = "tb",
        label = "TensorBoard",
        man = "mlr3torch::mlr_callback_set.tb"
    )
})
