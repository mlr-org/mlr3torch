#' @title LearnerTorchClassif
LearnerClassifTorch = R6Class("LearnerClassifTorch",
  inherit = LearnerClassifTorchAbstract,
  public = list(
    initialize = function(id = "classif.torch", param_vals = list(), .optimizer) {
      super$initialize(
        id = id,
        properties = c("twoclass", "multiclass", "hotstart_forward", "weights"),
        label = "Neural Network Classifier",
        feature_types = c("logical", "integer", "numeric", "factor"),
        .optimizer = .optimizer
      )
    }
  ),
  private = list(
    .train = function(task) {
      state = private$.build(task)
      learner_classif_torch_train(self, state, task)
    },
    .predict = function(task) {
      # When keep_last_prediction = TRUE we store the predictions of the last validation and we
      # therefore don't have to recompute them in the resample(), but can simple return the
      # cached predictions
      learner_classif_torch_predict(self, task)
    },
    .build = function(task) {
      build_torch(self, task)
    },
    .optimizer = NULL
  ),
  active = list(
    #' @field params ()
    parameters = function(rhs) {
      assert_ro_binding(rhs)
      self$state$model$network$parameters
    },
    history = function(rhs) {
      assert_ro_binding(rhs)
      self$state$model$history
    },
    optimizer = function(rhs) {
      assert_ro_binding(rhs)
      self$state$model$optimizer
    },
    criterion = function(rhs) {
      assert_ro_binding(rhs)
      self$state$model$criterion
    },
    network = function(rhs) {
      assert_ro_binding(rhs)
      self$state$model$network
    }
  )
)

mlr_learners$add("classif.torch", LearnerClassifTorch)


if (FALSE) {
  l = lrn("classif.torch", optimizer = "adam", criterion = "bce")
}
