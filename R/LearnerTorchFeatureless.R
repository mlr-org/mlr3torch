#' @title Featureless Torch Learner
#'
#' @templateVar name torch_featureless
#' @templateVar task_types classif, regr
#' @template params_learner
#' @template learner
#' @template learner_example
#'
#' @description
#' Featureless torch learner.
#' Output is a constant weight that is learned during training.
#' For classification, this should (asymptoptically) result in a majority class prediction when using the standard cross-entropy loss.
#' For regression, this should result in the median for L1 loss and in the mean for L2 loss.
#'
#' @section Parameters:
#' Only those from [`LearnerTorch`].
#'
#' @export
LearnerTorchFeatureless = R6Class("LearnerTorchFeatureless",
  inherit = LearnerTorch,
  public = list(
    #' @description Creates a new instance of this [R6][R6::R6Class] class.
    initialize = function(task_type, optimizer = NULL, loss = NULL, callbacks = list()) {
      super$initialize(
        id = paste0(task_type, ".torch_featureless"),
        task_type = task_type,
        label = "Featureless Torch Learner",
        param_set = ps(),
        properties = c("missings", "featureless"),
        feature_types = unname(mlr_reflections$task_feature_types),
        man = "mlr3torch::mlr_learners.torch_featureless",
        optimizer = optimizer,
        loss = loss,
        callbacks = callbacks
      )
    }
  ),
  private = list(
    .network = function(task, param_vals) {
      nn_featureless(nout = output_dim_for(task))
    },
    .dataset = function(task, param_vals) {
      dataset_featureless(task, cache_data = param_vals$cache_task_data %??% TRUE)
    }
  )
)

dataset_featureless = dataset(
  initialize = function(task, cache_data = TRUE) {
    self$task = task
    self$target_batchgetter = get_target_batchgetter(task)
    # like in task_dataset(), the target can be retrieved once instead of for every batch
    self$target = if (assert_flag(cache_data)) task$data(cols = task$target_names)
  },
  .getbatch = function(index) {
    target = if (is.null(self$target)) {
      self$task$data(rows = self$task$row_ids[index], cols = self$task$target_names)
    } else {
      self$target[index]
    }
    y = self$target_batchgetter(target)
    list(
      x = list(n = torch_tensor(nrow(target), dtype = torch_long())),
      y = y,
      .index = torch_tensor(index, dtype = torch_long())
    )

  },
  .length = function() {
    self$task$nrow
  }
)


nn_featureless = nn_module(
  initialize = function(nout) {
    self$weights = nn_parameter(torch_randn(nout))
    self$nout = nout
  },
  forward = function(n) {
    self$weights$expand(c(n$item(), self$nout))
  }
)

register_learner("classif.torch_featureless", LearnerTorchFeatureless)
register_learner("regr.torch_featureless", LearnerTorchFeatureless)
