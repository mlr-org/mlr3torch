#' @title TensorBoard Logging Callback
#'
#' @name mlr_callback_set.tb
#'
#' @description
#' Logs training loss, training measures, and validation measures as events.
#' To view them, use TensorBoard with `tensorflow::tensorboard()` (requires `tensorflow`) or the CLI.
#' @details
#' Logs events at most every epoch.
#'
#' @param path (`character(1)`)\cr
#'   The path to a folder where the events are logged.
#'   Point TensorBoard to this folder to view them.
#' @param log_train_loss (`logical(1)`)\cr
#'  Whether we log the training loss.
#' @family Callback
#' @export
#' @include CallbackSet.R
CallbackSetTB = R6Class("CallbackSetTB",
  inherit = CallbackSet,
  lock_objects = FALSE,
  public = list(
    #' @description
    #' Creates a new instance of this [R6][R6::R6Class] class.
    initialize = function(path, log_train_loss) {
      self$path = assert_path_for_output(path)
      if (!dir.exists(path)) {
        dir.create(path, recursive = TRUE)
      }
      self$log_train_loss = assert_flag(log_train_loss)
    },
    #' @description
    #' Logs the training loss, training measures, and validation measures as TensorBoard events.
    on_epoch_end = function() {
      if (self$log_train_loss) {
        private$.log_train_loss()
      }

      if (length(self$ctx$last_scores_train)) {
        walk(names(self$ctx$measures_train), private$.log_train_score)
      }

      if (length(self$ctx$last_scores_valid)) {
        walk(names(self$ctx$measures_valid), private$.log_valid_score)
      }
    }
  ),
  private = list(
    .log_score = function(prefix, measure_name, score) {
      event_list = set_names(list(score, self$ctx$epoch), c(paste0(prefix, measure_name), "step"))

      with_logdir(self$path, {
        do.call(log_event, event_list)
      })
    },
    .log_valid_score = function(measure_name) {
      valid_score = self$ctx$last_scores_valid[[measure_name]]
      private$.log_score("valid.", measure_name, valid_score)
    },
    .log_train_score = function(measure_name) {
      train_score = self$ctx$last_scores_train[[measure_name]]
      private$.log_score("train.", measure_name, train_score)
    },
    .log_train_loss = function() {
      with_logdir(self$path, {
        log_event(train.loss = self$ctx$last_loss)
      })
    }
  )
)

#' @include TorchCallback.R
mlr3torch_callbacks$add("tb", function() {
  TorchCallback$new(
    callback_generator = CallbackSetTB,
    param_set = ps(
      path           = p_uty(tags = c("train", "required")),
      log_train_loss = p_lgl(tags = c("train", "required"), init = FALSE)
    ),
    id = "tb",
    label = "TensorBoard",
    man = "mlr3torch::mlr_callback_set.tb"
  )
})
