#' @title TensorBoard Logging Callback
#'
#' @name mlr_callback_set.tb
#'
#' @description
#' Logs training loss, training measures, and validation measures as events.
#' To view them, use TensorBoard with `tensorflow::tensorboard()` (requires `tensorflow`) or the CLI.
#'
#' @section Resuming:
#' This callback keeps no state of its own.
#' The measures are logged under the epoch they belong to and the training loss under
#' [`ContextTorch`]'s `global_step`, both of which a resumed run continues counting rather than
#' restarting, so its curves extend those of the run it continues.
#' Point the resumed run at the `path` its predecessor wrote and both halves end up in one
#' TensorBoard run; a fresh `path` puts them in two.
#'
#' @details
#' Logs events at most every epoch.
#'
#' @param path (`character(1)`)\cr
#'   The path to a folder where the events are logged.
#'   Point TensorBoard to this folder to view them.
#'   The folder must be new, empty, or one that this callback already logged into, so that a
#'   resumed run can continue the log of the run it continues.
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
      # A folder that already holds events is accepted so that a resumed run can log into the one
      # the run it continues wrote, which is what keeps both halves in a single TensorBoard run.
      # A folder holding anything else is still refused, so no unrelated data is written into.
      self$path = if (is_empty_dir(path) || length(list.files(path, pattern = "^events\\.out\\.tfevents\\."))) {
        path
      } else {
        assert_path_for_output(path)
      }
      if (!dir.exists(path)) {
        dir.create(path, recursive = TRUE)
      }
      self$log_train_loss = assert_flag(log_train_loss)
      if (self$log_train_loss) {
        self$on_batch_end = function() {
          private$.log_train_loss()
        }
      }
    },
    #' @description
    #' Logs the training loss, training measures, and validation measures as TensorBoard events.
    on_epoch_end = function() {
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

      tfevents::with_logdir(self$path, {
        do.call(tfevents::log_event, event_list)
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
      tfevents::with_logdir(self$path, {
        tfevents::log_event(train.loss = self$ctx$last_loss, step = self$ctx$global_step)
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
    packages = "tfevents",
    label = "TensorBoard",
    man = "mlr3torch::mlr_callback_set.tb"
  )
})
