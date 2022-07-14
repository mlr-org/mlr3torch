#' @title LearnerTorchClassif
#'
#' @name mlr_learners_classif.torch
#'
#' @template param_id
#' @template param_param_vals
#' @template param_optimizer
#' @template param_loss
#' @param feature_types (`character()`)\cr
#'   The feature types the learner supports. The default is all feature types.
#' @param network (`nn_module()`)\cr
#'   An object of class `"nn_module"` as defined in `torch`.
#' @export
LearnerClassifTorch = R6Class("LearnerClassifTorch",
  inherit = LearnerClassifTorchAbstract,
  public = list(
    #' @description Initializes an instance of this [R6][R6::R6Class] class.
    initialize = function(param_vals = list(), optimizer, loss, network,
      feature_types = NULL) {
      if (inherits(network, "nn_Module")) {
        stopf("The network must be initialized by calling the function (and not with '$new()').")
      }
      assert(check_function(network), check_class(network, "nn_module"))
      private$..network = network
      param_set = ps()

      super$initialize(
        id = "classif.torch",
        properties = c("weights", "twoclass", "multiclass", "hotstart_forward"),
        label = "Classification Network",
        feature_types = feature_types %??% mlr_reflections$task_feature_types,
        optimizer = optimizer,
        loss = loss,
        man = "mlr3torch::mlr_learners_classif.torch",
        param_set = param_set
      )
    }
  ),
  private = list(
    .network = function(task) {
      private$..network
    },
    ..network = NULL
  )
)
