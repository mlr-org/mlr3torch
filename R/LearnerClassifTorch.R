LearnerClassifTorch = R6Class("LearnerClassifTorch",
  inherit = LearnerTorch,
  public = list(
    initialize = function() {
      # TODO: modify input checks to classification
      super$initialize(
        task_type = "classif",
        predict_types = c("response"),
        param_set = dl_paramset(),
        properties = c("twoclass", "multiclass")
      )
    }
  )
)

if (FALSE) {
  learner = LearnerClassifTorch$new()
}
