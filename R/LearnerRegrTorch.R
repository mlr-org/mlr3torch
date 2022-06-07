#' @title LearnerRegrTorch
#' @export
LearnerRegrTorch = R6Class("LearnerRegrTorch",
  inherit = LearnerRegrTorchAbstract,
  public = list(
    #' @description Initializes an instance of this [R6][R6::R6Class] class.
    #' @param id (`character(1)`)\cr
    #'   The id for of the object.
    #' @param param_vals (named `list()`)\cr
    #'   The initial parameters for the object.
    #' @param .optimizer (`character(1)`)\cr
    #'   The optimizer, see `torch_reflections$optimizer`.
    #' @param .loss (`character(1)`)\cr
    #'   The loss, see `torch_reflections$loss`.
    #' @param .feature_types (`character()`)\cr
    #'   The feature types the learner supports. The default is all feature types.
    initialize = function(id = "classif.torch", param_vals = list(), .optimizer, .loss,
      .feature_types = NULL) {
      super$initialize(
        id = id,
        properties = c("weights", "hotstart_forward"),
        label = "Classification Network",
        feature_types = .feature_types %??% mlr_reflections$task_feature_types,
        optimizer = .optimizer,
        loss = .loss,
        man = "mlr3torch::mlr_learners_classif.torch"
      )
    }
  ),
  private = list(
    .network = function(task) {
      architecture = self$param_set$values$architecture

      if (test_r6(architecture, "Graphitecture")) {
        network = architecture$build(task)
      } else if (test_r6(architecture, "nn_Module")) {
        network = architecture$clone(deep = TRUE)
      } else {
        stopf("Invalid argument for architecture.")
      }
    }
  )
)
