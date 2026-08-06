#' @title Checkpoint Callback
#'
#' @name mlr_callback_set.checkpoint
#'
#' @description
#' Saves the optimizer and network states during training.
#' The final network and optimizer are always stored.
#'
#' Checkpoints are written at the end of an epoch. For one written after epoch `<n>`, two files
#' are created in `path`:
#' * `network<n>.pt` :: The `$state_dict()` of the network.
#' * `optimizer<n>.pt` :: The `$state_dict()` of the optimizer.
#'
#' An epoch that was interrupted -- because training failed or was stopped -- is not written under
#' its own number, so `network<n>.pt` is always the network at the *end* of epoch `n`.
#'
#' @section Checkpointing more than one training run:
#' So that a checkpoint never overwrites unrelated data, `path` must not exist yet or must be an
#' empty directory. A fixed `path` can therefore only be used for a *single* training run: training
#' the same learner a second time into it fails, and so do `resample()`, `benchmark()` and tuning,
#' whose second iteration would write into the directory the first one filled.
#'
#' Pass a `function()` as `path` to check-point such a run. It is called once at the beginning of
#' every training run and must return the path for that run, so each iteration gets its own
#' directory:
#'
#' ```r
#' root = tempfile()
#' learner$param_set$set_values(
#'   cb.checkpoint.path = function() tempfile(tmpdir = root)
#' )
#' ```
#'
#' Note that the iterations of a `resample()` are not distinguishable from within the callback, so
#' the function has to make the paths unique itself, e.g. via [`tempfile()`] as above. When training
#' happens in parallel, the function runs in the worker, so it must not rely on state of the main
#' session.
#'
#' @details
#' Saving the learner itself in the callback with a trained model is impossible,
#' as the model slot is set *after* the last callback step is executed.
#'
#' @param path (`character(1)` | `function()`)\cr
#'   The path to a folder where the models are saved, or a function of no arguments returning it.
#'   The folder must be new or empty, see section *Checkpointing more than one training run*.
#' @param freq (`integer(1)`)\cr
#'   How often the model is saved, in epochs.
#' @family Callback
#' @export
#' @include CallbackSet.R
#' 
#' @examplesIf torch::torch_is_installed()
#' cb = t_clbk("checkpoint", freq = 1)
#' task = tsk("iris")
#' 
#' pth = tempfile()
#' learner = lrn("classif.mlp", epochs = 3, batch_size = 1, callbacks = cb)
#' learner$param_set$set_values(cb.checkpoint.path = pth)
#' 
#' learner$train(task)
#' 
#' list.files(pth)
CallbackSetCheckpoint = R6Class("CallbackSetCheckpoint",
  inherit = CallbackSet,
  lock_objects = FALSE,
  # TODO: This should also save the learner itself
  public = list(
    #' @field weight (`numeric(1)`)\cr
    #'   `Inf`, so that this callback runs after the other callbacks and hence saves the network and
    #'   optimizer as they are at the end of the stage, see section *Ordering* of [`CallbackSet`].
    #'   The only exception is the restore of `restore_best_weights`, which happens afterwards, so
    #'   a checkpoint always holds the network as training left it.
    weight = Inf,
    #' @description
    #' Creates a new instance of this [R6][R6::R6Class] class.
    initialize = function(path, freq) {
      self$freq = assert_int(freq, lower = 1L)
      # a function is evaluated once per training run, which is what makes resampling, benchmarking
      # and tuning possible: every iteration can compute a path of its own
      if (is.function(path)) {
        path = assert_string(path(), .var.name = "return value of the `path` function")
      }
      # an empty folder is a legal target: that is what a pre-created output directory looks like,
      # and what a run that died before writing its first checkpoint leaves behind. Any other
      # existing path is rejected, so that unrelated data is never written into.
      self$path = if (is_empty_dir(path)) path else assert_path_for_output(path)
      if (!dir.exists(path)) {
        dir.create(path, recursive = TRUE)
      }
    },
    #' @description
    #' Saves the network and optimizer state dict.
    #' Does nothing if `freq` is not met.
    on_epoch_end = function() {
      # tracked even when nothing is saved here, so that the 'exit' stage knows how many epochs
      # were actually completed
      private$.complete_epochs = self$ctx$epoch
      if (self$ctx$epoch %% self$freq != 0) {
        return(NULL)
      }
      private$.save(self$ctx$epoch)
    },
    #' @description
    #' Saves the final network and optimizer, unless the last complete epoch was already saved.
    on_exit = function() {
      # this stage also runs when training was interrupted, in which case the epoch in progress is
      # deliberately not saved: `network<n>.pt` is meant to be the network at the *end* of epoch n,
      # and the weights of a half-trained epoch are not that.
      if (private$.complete_epochs == 0L || private$.complete_epochs %% self$freq == 0) {
        # nothing was completed, or the last complete epoch was already saved
        return(NULL)
      }
      private$.save(private$.complete_epochs)
    }
  ),
  private = list(
    # the number of epochs that were trained to completion
    .complete_epochs = 0L,
    .save = function(suffix) {
      torch_save(self$ctx$network$state_dict(), file.path(self$path, paste0("network", suffix, ".pt")))
      torch_save(self$ctx$optimizer$state_dict(), file.path(self$path, paste0("optimizer", suffix, ".pt")))
    }
  )
)

# Whether `path` exists and holds nothing.
is_empty_dir = function(path) {
  dir.exists(path) && !length(list.files(path, all.files = TRUE, no.. = TRUE))
}

# the path is either given directly or computed once per training run
check_checkpoint_path = function(x) {
  if (is.function(x)) check_function(x, nargs = 0L) else check_string(x)
}

#' @include TorchCallback.R
mlr3torch_callbacks$add("checkpoint", function() {
  TorchCallback$new(
    callback_generator = CallbackSetCheckpoint,
    param_set = ps(
      path = p_uty(tags = c("train", "required"), custom_check = check_checkpoint_path),
      freq = p_int(lower = 1L, tags = c("train", "required"))
    ),
    id = "checkpoint",
    label = "Checkpoint",
    man = "mlr3torch::mlr_callback_set.checkpoint"
  )
})
