#' @title Abstract Base Class for Torch Classification Network
#' @description
#' All Torch Classification Learners should inherit from this base class.
#' It implements basic functionality that can be reused for all sort of learners
#' It is not intended for direct use.
#' @parameter
#'
#' @export
LearnerClassifTorchAbstract = R6Class("LearnerClassifTorchAbstract",
  inherit = LearnerClassif,
  public = list(
    #' @description Initializes an object.
    #' The parameters of the learner are constructed dynamically, depending on the choice of the
    #' `optimizer`.
    #' @param id (`character(1)`) The id of the learner.
    #' @param param_vals (`list()`) Parameters values to be set.
    #' @param param_set (`paradox::ParamSet`) Additional parameters to the standard paramset.
    #' @param .optimizer (`character(1)`) The name of the optimizer.
    #' @param .loss (`character(1)` || `nn_loss`).
    initialize = function(id, .optimizer, .loss, param_set = ps(), label = NULL, properties = NULL,
      packages = character(0), predict_types = NULL, feature_types, preprocessing) {
      private$.optimizer = .optimizer
      private$.loss = .loss
      # FIXME: loglik?
      if (is.null(properties)) {
        properties = c("weights", "multiclass", "twoclass", "hotstart_forward")
      }
      if (is.null(predict_types)) {
        predict_types = "response"
      }
      packages = assert_character(packages, any.missing = FALSE, min.chars = 1L)
      packages = union(c("mlr3torch", "torch"), packages)
      # note that we don't have to explicitly check that the optimizer params are disjunct from
      # the remaining parameters as this is done here anyway (call fails if it doesn't).
      param_set_complete = make_paramset("classif", .optimizer, .loss, architecture = TRUE)
      param_set_complete$add(param_set)

      super$initialize(
        id = id,
        packages = packages,
        param_set = param_set_complete,
        predict_types = predict_types,
        properties = properties,
        data_formats = "data.table",
        label = label,
        feature_types = feature_types
      )
    }
  ),
  private = list(
    .train = function(task) {
      stop("ABC")
    },
    .predict = function(task) {
      stop("ABC")
    },
    .optimizer = NULL,
    .loss = NULL
  ),
  active = list(
    #' @field params ()
    parameters = function(rhs) {
      stop("ABC")
    },
    history = function(rhs) {
      stop("ABC")
    },
    optimizer = function(rhs) {
      stop("ABC")
    },
    loss = function(rhs) {
      stop("ABC")
    },
    network = function(rhs) {
      stop("ABC")
    }
  )
)
