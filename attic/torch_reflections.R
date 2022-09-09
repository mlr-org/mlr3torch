#' @title Reflections mechanism for torch
#'
#' @details
#' Used to store / extend available hyperparameter levels for options used throughout torch,
#' e.g. the available 'loss' for a given Learner.
#'
#' @format [environment].
#' @export
torch_reflections = new.env(parent = emptyenv())

torch_reflections$callback_steps = c(
  "start",
  "before_train_epoch",
  "before_train_batch",
  "after_train_batch",
  "before_valid_epoch",
  "before_valid_batch",
  "after_valid_batch",
  "after_valid_epoch",
  "end"
)

