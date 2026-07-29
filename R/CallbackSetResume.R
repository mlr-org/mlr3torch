#' @title Resume Callback
#'
#' @name mlr_callback_set.resume
#'
#' @description
#' Continues training from a checkpoint, also in a new R session.
#' The network and optimizer states are loaded before the first epoch, and training then continues
#' until the learner's `epochs` are reached.
#'
#' The `epochs` parameter of the learner is the *total* number of epochs, i.e. it includes the
#' epochs that were already trained.
#' Resuming a checkpoint that was written after 5 epochs with a learner that is configured with
#' `epochs = 8` therefore trains 3 more epochs.
#' Nothing is trained if the checkpoint is already at (or beyond) the configured `epochs`.
#'
#' @section Which checkpoint is used:
#' Usually, `path` is set to the folder that a [`CallbackSetCheckpoint`] wrote to and the most
#' recent checkpoint in it is used.
#' If that folder contains no checkpoint -- as in the first run of a script that is meant to be
#' restarted -- training simply starts from scratch, so that the same script can be used to start
#' and to continue a run.
#'
#' Alternatively, `network_path` and `optimizer_path` can point to specific files, which is useful
#' for checkpoints that were not written by [`CallbackSetCheckpoint`].
#' These files must exist.
#'
#' @section Callback States:
#' Checkpoints written by [`CallbackSetCheckpoint`] also contain the states of the other callbacks,
#' which are restored into the callbacks of the same id.
#' This means that e.g. the training history of the previous run is preserved.
#' Callbacks are executed in the order in which they are passed to the learner, so this callback
#' should be passed *first* if another callback overwrites its own state in its `on_begin()` stage.
#'
#' Callbacks that do not implement `$load_state_dict()` are skipped with a warning, as are the
#' states of callbacks that are not part of the current training run.
#'
#' Note that schedules which are defined over the *total* number of steps -- most notably
#' [`mlr_callback_set.lr_scheduler_one_cycle`] -- are restored as they were configured in the
#' interrupted run.
#' Configure the resuming run with the same `epochs` as the original one, otherwise the restored
#' schedule and the number of remaining steps disagree; this is checked before the first epoch is
#' trained and raises an error.
#'
#' @param path (`character(1)` or `NULL`)\cr
#'   Path to a folder that a [`CallbackSetCheckpoint`] wrote to.
#'   The most recent checkpoint in this folder is used.
#'   If the folder does not exist or contains no checkpoint, training starts from scratch.
#' @param network_path (`character(1)` or `NULL`)\cr
#'   Path to a file containing the `$state_dict()` of a network, as saved by [`torch::torch_save()`].
#'   Overrides the network found via `path`.
#' @param optimizer_path (`character(1)` or `NULL`)\cr
#'   Path to a file containing the `$state_dict()` of an optimizer.
#'   Overrides the optimizer found via `path`.
#' @param epochs_trained (`integer(1)` or `NULL`)\cr
#'   The number of epochs that the checkpoint was trained for.
#'   Only needs to be set when it cannot be read from the checkpoint, which is the case for
#'   checkpoints that were not written by [`CallbackSetCheckpoint`].
#'   Default is `NULL`.
#'
#' @family Callback
#' @export
#' @include CallbackSet.R
#'
#' @examplesIf torch::torch_is_installed()
#' task = tsk("iris")
#' pth = tempfile()
#'
#' # train for 2 epochs, checkpointing after every epoch
#' learner = lrn("classif.mlp", epochs = 2, batch_size = 50,
#'   callbacks = t_clbk("checkpoint", freq = 1))
#' learner$param_set$set_values(cb.checkpoint.path = pth)
#' learner$train(task)
#'
#' # continue until epoch 5, i.e. train 3 more epochs
#' learner_resumed = lrn("classif.mlp", epochs = 5, batch_size = 50,
#'   callbacks = t_clbk("resume"))
#' learner_resumed$param_set$set_values(cb.resume.path = pth)
#' learner_resumed$train(task)
#' learner_resumed$model$epochs
CallbackSetResume = R6Class("CallbackSetResume",
  inherit = CallbackSet,
  lock_objects = FALSE,
  public = list(
    #' @description
    #' Creates a new instance of this [R6][R6::R6Class] class.
    initialize = function(path = NULL, network_path = NULL, optimizer_path = NULL,
      epochs_trained = NULL) {
      self$path = assert_string(path, null.ok = TRUE)
      self$network_path = assert_string(network_path, null.ok = TRUE)
      self$optimizer_path = assert_string(optimizer_path, null.ok = TRUE)
      self$epochs_trained = assert_int(epochs_trained, lower = 0L, null.ok = TRUE)

      if (is.null(path) && is.null(network_path) && is.null(optimizer_path)) {
        stopf("Either 'path' or 'network_path' and 'optimizer_path' must be provided.")
      }
      if (xor(is.null(network_path), is.null(optimizer_path))) {
        stopf("'network_path' and 'optimizer_path' must either both be provided or both be NULL.")
      }
      walk(discard(list(network_path, optimizer_path), is.null), function(p) {
        if (!file.exists(p)) stopf("File '%s' does not exist.", p)
      })
    },
    #' @description
    #' Loads the network, optimizer and callback states and continues training from the epoch the
    #' checkpoint was written in.
    on_begin = function() {
      paths = private$.checkpoint_paths()
      if (is.null(paths)) {
        lg$info("No checkpoint found in '%s', starting training from scratch.", self$path)
        return(NULL)
      }

      self$ctx$network$load_state_dict(torch_load(paths$network))
      self$ctx$optimizer$load_state_dict(torch_load(paths$optimizer))

      state = if (!is.null(paths$state)) readRDS(paths$state)
      epochs_trained = self$epochs_trained %??% state$epoch %??% paths$epochs_fallback
      private$.load_callback_states(state$callbacks)

      if (epochs_trained >= self$ctx$total_epochs) {
        warningf(paste0("The checkpoint was trained for %i epochs, but the learner is configured ",
          "with epochs = %i. No further epochs are trained. Note that 'epochs' is the total ",
          "number of epochs, including those of the checkpoint."),
          epochs_trained, self$ctx$total_epochs)
      }
      self$ctx$epoch = epochs_trained
    }
  ),
  private = list(
    # the network, optimizer and state files to resume from, or NULL if there is nothing to resume
    .checkpoint_paths = function() {
      if (!is.null(self$network_path)) {
        # nothing tells us how far such a checkpoint was trained, so unless `epochs_trained` says
        # otherwise we treat it as a warm start and train the full number of epochs
        return(list(network = self$network_path, optimizer = self$optimizer_path,
          state = NULL, epochs_fallback = 0L))
      }
      suffixes = checkpoint_suffixes(self$path)
      if (!length(suffixes)) return(NULL)

      suffix = suffixes[1L]
      state = file.path(self$path, paste0("state", suffix, ".rds"))
      list(
        network   = file.path(self$path, paste0("network", suffix, ".pt")),
        optimizer = file.path(self$path, paste0("optimizer", suffix, ".pt")),
        state     = if (file.exists(state)) state,
        # without a state file (checkpoints written by earlier versions) the suffix is the best
        # guess, which is correct as long as the checkpoint used freq_type = "epoch"
        epochs_fallback = suffix
      )
    },
    .load_callback_states = function(states) {
      if (!length(states)) return(NULL)
      unknown = setdiff(names(states), names(self$ctx$callbacks))
      if (length(unknown)) {
        warningf("The checkpoint contains states for callback(s) %s, which are not part of this training run. They are ignored.", # nolint
          paste0("'", unknown, "'", collapse = ", "))
      }
      iwalk(states[intersect(names(states), names(self$ctx$callbacks))], function(state, id) {
        cb = self$ctx$callbacks[[id]]
        # the default method only accepts NULL, i.e. the callback cannot restore anything.
        # R6 binds methods to the object's environment, so only the body can be compared.
        if (identical(body(cb$load_state_dict), body(CallbackSet$public_methods$load_state_dict))) {
          warningf("Callback '%s' does not implement $load_state_dict(), its state is ignored.", id)
          return(NULL)
        }
        cb$load_state_dict(state)
      })
    }
  )
)

#' @include TorchCallback.R
mlr3torch_callbacks$add("resume", function() {
  TorchCallback$new(
    callback_generator = CallbackSetResume,
    param_set = ps(
      path           = p_uty(default = NULL, tags = "train"),
      network_path   = p_uty(default = NULL, tags = "train"),
      optimizer_path = p_uty(default = NULL, tags = "train"),
      epochs_trained = p_int(lower = 0L, default = NULL, special_vals = list(NULL), tags = "train")
    ),
    id = "resume",
    label = "Resume",
    man = "mlr3torch::mlr_callback_set.resume"
  )
})
