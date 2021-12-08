#' @title Regression TabNet Learner
#' @author Lukas Burk
#' @name mlr_learners_regr.torch.tabnet
#'
#' @template class_learner
#' @templateVar id regr.torch.tabnet
#' @templateVar caller tabnet
#'
#' @references
#' <FIXME - DELETE THIS AND LINE ABOVE IF OMITTED>
#'
#' @template seealso_learner
#' @export
#' @examples
#' \dontrun{
#' library(mlr3)
#' library(mlr3torch)
#'
#' task <- tsk("boston_housing")
#'
#' # Creating a learner & training on example task
#' lrn <- lrn("regr.torch.tabnet")
#'
#' lrn$param_set$values$epochs <- 10
#' lrn$train(task)
#'
#' # Predict on training data, get RMSE
#' predictions <- lrn$predict(task)
#' predictions$score(msr("regr.rmse"))
#' }
LearnerRegrTorchTabnet = R6::R6Class("LearnerRegrTorchTabnet",
  inherit = LearnerRegr,

  public = list(
    #' @description
    #' Creates a new instance of this [R6][R6::R6Class] class.
    initialize = function() {
      ps = params_tabnet()

      super$initialize(
        id = "regr.torch.tabnet",
        packages = "tabnet",
        feature_types = c("logical", "integer", "numeric", "factor", "ordered"),
        param_set = ps,
        properties = c("importance", "missings", "selected_features"),
        man = "mlr3torch::mlr_learners_regr.torch.tabnet"
      )
    }

    # FIXME - ADD IMPORTANCE METHOD HERE AND DELETE THIS LINE.
    # <See LearnerRegrRandomForest for an example>
    # @description
    # The importance scores are extracted from the slot <FIXME>.
    # @return Named `numeric()`.
    # importance = function() { }

  ),

  private = list(

    .train = function(task) {
      # get parameters for training
      pars = self$param_set$get_values(tags = "train")

      # Drop control par from training pars as tabnet_fit doesn't know it
      pars <- pars[!(names(pars) %in% names(pars_control))]

      # set column names to ensure consistency in fit and predict
      self$state$feature_names = task$feature_names

      # Create objects for the train call
      formula = task$formula()
      data = task$data()

      # use the mlr3misc::invoke function (it's similar to do.call())
      mlr3misc::invoke(tabnet::tabnet_fit,
                       formula = formula,
                       data = data,
                       .args = pars)
    },

    .predict = function(task) {
      # get parameters with tag "predict"
      pars = self$param_set$get_values(tags = "predict")

      # get newdata and ensure same ordering in train and predict
      newdata = task$data(cols = self$state$feature_names)


      pred = mlr3misc::invoke(predict, self$model, new_data = newdata,
                              .args = pars)

      list(response = pred[[".pred"]])
    }

  )
)
