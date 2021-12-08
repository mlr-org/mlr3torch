#' @title Classification TabNet Learner
#' @author Lukas Burk
#' @name mlr_learners_classif.torch.tabnet
#'
#' @template class_learner
#' @templateVar id classif.torch.tabnet
#' @templateVar caller tabnet
#'
#'
#' @template seealso_learner
#' @export
#' @examples
#' \dontrun{
#' library(mlr3)
#' library(mlr3torch)
#' task <- tsk("german_credit")
#' lrn <- lrn("classif.torch.tabnet")
#'
#' lrn$param_set$values$epochs <- 10
#' lrn$param_set$values$attention_width <- 8
#' lrn$train(task)
#' }
LearnerClassifTorchTabnet = R6::R6Class("LearnerClassifTorchTabnet",
  inherit = LearnerClassif,

  public = list(
    #' @description
    #' Creates a new instance of this [R6][R6::R6Class] class.
    initialize = function() {
      ps = params_tabnet()

      super$initialize(
        id = "classif.torch.tabnet",
        packages = "tabnet",
        feature_types = c("logical", "integer", "numeric", "factor", "ordered"),
        predict_types = c("response", "prob"),
        param_set = ps,
        properties = c("importance", "missings", "multiclass", "selected_features", "twoclass", "weights"),
        man = "mlr3torch::mlr_learners_classif.torch.tabnet"
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
      pars_control = self$param_set$get_values(tags = "control")

      # Drop control par from training pars as tabnet_fit doesn't know it
      pars <- pars[!(names(pars) %in% names(pars_control))]

      # Set number of threads
      torch::torch_set_num_threads(pars_control$num_threads)

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

      if (self$predict_type == "response") {
        pred = mlr3misc::invoke(predict, self$model, new_data = newdata,
                                type = "class", .args = pars)

        list(response = pred[[".pred_class"]])
      } else {
        pred = mlr3misc::invoke(predict, self$model, new_data = newdata,
                                type = "prob", .args = pars)

        # Result will be a df with one column per variable with names '.pred_<level>'
        # we want the names without ".pred"
        names(pred) <- sub(pattern = ".pred_", replacement = "", names(pred))

        list(prob = as.matrix(pred))
      }

    }
  )
)
