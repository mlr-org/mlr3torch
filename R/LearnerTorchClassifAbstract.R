#' @title Abstract Base Class for Classification Torch Learner
#' @description
#' All Torch Classification Learner inherit from this class
#' It is not intended to be used directly by the user.
#' @export
LearnerTorchClassifAbstract = R6Class("LearnerTorchClassifAbstract",
  inherit = LearnerClassif,
  public = list(
    optimizer = NULL,
    criterion = NULL,
    initialize = function(id, param_vals, param_set = ps(), .optimizer = NULL, .criterion = NULL) {
      self$optimizer = .optimizer
      self$criterion = .criterion

      param_set = c(
        ps(
          n_epochs = p_int(tags = "train", lower = 0L),
          device = p_fct(tags = c("train", "predict"), levels = c("cpu", "cuda"), default = "cpu"),
          batch_size = p_int(tags = c("train", "predict"), lower = 1L, default = 16L)
        ),
        param_set
      )
      if (!is.null(.optimizer)) {
        param_set = c(param_set, get_paramset_optim(.optimizer))
      }
      if (!is.null(.criterion)) {
        param_set = c(param_set, get_paramset_optim(.criterion))
      }

      super$initialize(
        id = id,
        param_set = param_set,
        param_vals = param_vals
      )
    }
  )
)
