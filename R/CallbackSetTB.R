#' @title TensorBoard Logging Callback
#'
#' @name mlr_callback_set.tb
#'
#' @description
#' Logs training loss and validation measures as events that can be tracked using TensorBoard.
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
    lock_objects = FALSE,
    public = list(
        path = NULL,
        #' @description
        #' Creates a new instance of this [R6][R6::R6Class] class.
        initialize = function(path) {
          self$path = assert_path_for_output(path)
          if (!dir.exists(path)) {
            dir.create(path, recursive = TRUE)
          }
        },
        #' @description
        #' Logs the training loss and validation measures as TensorFlow events.
        #' Meaningful changes happen at the end of each epoch.
        #' Notably NOT on_batch_valid_end, since there are no gradient steps between validation batches,
        #' and therefore differences are due to randomness
        # TODO: display the appropriate x axis with its label in TensorBoard
        # relevant when we log different scores at different times
        on_epoch_end = function() {
            log_valid_score = function(measure_name) {
                valid_score = list(self$ctx$last_scores_valid[[measure_name]])
                names(valid_score) = paste0("valid.", measure_name)
                with_logdir(self$path, {
                    do.call(log_event, valid_score)
                })
            }

            log_train_score = function(measure_name) {
                # TODO: figure out what self$ctx$last_loss looks like when there are multiple train measures
                # TODO: remind ourselves why we wanted to display last_loss and not last_scores_train
                with_logdir(self$path, {
                    log_event(train.loss = self$ctx$last_loss)
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
