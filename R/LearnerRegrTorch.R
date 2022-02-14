LearnerRegrTorch = R6Class("LearnerRegrTorch",
  inherit = LearnerTorch,
  public = list(
    initialize = function() {
      # TODO: modify nput checks to classification
      super$initialize(
        task_type = "regr",
        predict_types = c("response"),
        param_set = dl_paramset(),
        properties = c()
      )
    }
  )
)

