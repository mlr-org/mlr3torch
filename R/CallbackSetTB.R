#' @title TensorBoard Logging Callback
#'
#' @name mlr_callback_set.tb
#'
#' @description
#' Logs training loss, training measures, and validation measures as events.
#' To view them, use TensorBoard with `tensorflow::tensorboard()` (requires `tensorflow`) or the CLI.
#'
#' @section Resuming:
#' The measures are logged under the epoch they belong to, which a resumed run continues counting on
#' its own. The training loss is logged per batch and has no epoch to be logged under, so the step it
#' was last written at is stored in the checkpoint and restored, and a resumed run extends that curve
#' rather than writing a second one over it starting at `0`.
#'
#' A resumed run always logs into a folder of its own, as `path` must not exist yet.
#' Pointing TensorBoard at the folder that holds both then shows one continuous curve per measure
#' instead of two that overlap from the start.
#'
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
      if (self$log_train_loss) {
        self$on_batch_end = function() {
          private$.log_train_loss()
        }
      }
    },
    #' @description
    #' Returns the number of training losses logged so far, which is the step the next one is
    #' logged under.
    #' The measures do not need one, as they are logged under the epoch they belong to.
    state_dict = function() {
      list(batch_step = private$.batch_step)
    },
    #' @description
    #' Loads the step that the training loss continues at.
    #' @param state_dict (named `list()`)\cr
    #'   The state dict as retrieved via `$state_dict()`.
    load_state_dict = function(state_dict) {
      private$.batch_step = state_dict$batch_step
      invisible(NULL)
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
    # the step the next training loss is logged under. The measures are logged under `ctx$epoch`,
    # which a resumed run continues on its own, but the loss is logged per batch and so needs a
    # counter of its own. It is kept here rather than derived from `(epoch - 1) * batches + step`
    # so that it stays monotonic when a resumed run uses a different `batch_size`.
    .batch_step = 0L,
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
      # without an explicit step, tfevents counts from 0 in every run, so a resumed run would write
      # its loss curve over the one of the run it continues instead of extending it
      tfevents::with_logdir(self$path, {
        tfevents::log_event(train.loss = self$ctx$last_loss, step = private$.batch_step)
      })
      private$.batch_step = private$.batch_step + 1L
      invisible(NULL)
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
