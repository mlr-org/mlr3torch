#' @title Regression TabNet Learner
#' @author Lukas Burk
#' @name regr.torch.tabnet
#'
# @template class_learner
# @templateVar id regr.torch.tabnet
# @templateVar caller tabnet
#'
#' @references
#' <FIXME - DELETE THIS AND LINE ABOVE IF OMITTED>
#'
# @template seealso_learner
# @template example
#' @export
#' @examples
#' \dontrun{
#' library(mlr3)
#' library(mlr3torch)
#' # Creating a learner & training on example task
#' lrn = LearnerRegrTorchTabnet$new()
#' lrn$param_set$values$epochs = 10
#' lrn$train(tsk("boston_housing"))
#'
#' # Predict on training data, get RMSE
#' predictions <- lrn$predict(tsk("boston_housing"))
#' predictions$score(msr("regr.rmse"))
#' }
LearnerRegrTorchTabnet = R6::R6Class("LearnerRegrTorchTabnet",
  inherit = LearnerRegr,

  public = list(
    #' @description
    #' Creates a new instance of this [R6][R6::R6Class] class.
    initialize = function() {
      ps = ParamSet$new(list(
        ParamInt$new("batch_size", default = 256L, lower = 1L, upper = Inf, tags = "train"),
        ParamDbl$new("penalty",    default = 0.001, tags = "train"),

        # FIXME: NULL here is used for bool FALSE, not sure what to do there.
        ParamUty$new("clip_value",         default = NULL, tags = "train"),
        ParamFct$new("loss",               default = "auto", levels = c("auto", "mse", "cross_entropy"), tags = "train"),
        ParamInt$new("epochs",             default = 5L,  lower = 1L, upper = Inf, tags = "train"),
        ParamLgl$new("drop_last",          default = FALSE, tags = "train"),
        ParamInt$new("decision_width",     default = 8L, lower = 1L, upper = Inf, tags = "train"),
        ParamInt$new("attention_width",    default = 8L, lower = 1L, upper = Inf, tags = "train"),
        ParamInt$new("num_steps",          default = 3L,  lower = 1L, upper = Inf, tags = "train"),
        ParamDbl$new("feature_reusage",    default = 1.3, lower = 0, upper = Inf, tags = "train"),
        ParamFct$new("mask_type",          default = "sparsemax", levels = c("sparsemax", "entmax"), tags = "train"),
        ParamInt$new("virtual_batch_size", default = 128L, lower = 1L, upper = Inf, tags = "train"),
        ParamDbl$new("valid_split",        default = 0, lower = 0, upper = 1, tags = "train"),
        ParamDbl$new("learn_rate",         default = 0.02, lower = 0, upper = 1, tags = "train"),

        # FIXME: Currently either 'adam' or arbitrary optimizer function according to docs
        ParamUty$new("optimizer", default = "adam", tags = "train"),

        # FIXME: This is either NULL or a function or explicit "steps", needs custom_check fun
        ParamUty$new("lr_scheduler", default = NULL, tags = "train"),

        ParamDbl$new("lr_decay",          default = 0.1, lower = 0, upper = 1, tags = "train"),
        ParamInt$new("step_size",         default = 30L, lower = 1L, upper = Inf, tags = "train"),
        ParamInt$new("checkpoint_epochs", default = 10L, lower = 1L, upper = Inf, tags = "train"),
        ParamInt$new("cat_emb_dim",       default = 1L, lower = 0L, upper = Inf, tags = "train"),
        ParamInt$new("num_independent",   default = 2L, lower = 0, upper = Inf, tags = "train"),
        ParamInt$new("num_shared",        default = 2L, lower = 0, upper = Inf, tags = "train"),
        ParamDbl$new("momentum",          default = 0.02, lower = 0, upper = 1, tags = "train"),
        ParamDbl$new("pretraining_ratio", default = 0.5, lower = 0, upper = 1, tags = "train"),
        ParamLgl$new("verbose",           default = FALSE, tags = "train"),
        ParamFct$new("device",            default = "auto", levels = c("auto", "cpu", "cuda"), tags = "train")
        # ParamDbl$new("importance_sample_size", lower = 0, upper = 1, tags = "train"),
      ))

      # FIXME - MANUALLY UPDATE PARAM VALUES BELOW IF APPLICABLE THEN DELETE THIS LINE.
      ps$values = list(
        batch_size = 256,
        penalty = 0.001,
        #clip_value = NULL,
        loss = "auto",
        epochs = 5,
        drop_last = FALSE,
        decision_width = 8L,
        attention_width = 8L,
        num_steps = 3,
        feature_reusage = 1.3,
        mask_type = "sparsemax",
        virtual_batch_size = 128,
        valid_split = 0,
        learn_rate = 0.02,
        optimizer = "adam",
        #lr_scheduler = NULL,
        #lr_decay = 0.1,
        #step_size = 30,
        checkpoint_epochs = 10,
        cat_emb_dim = 1,
        num_independent = 2,
        num_shared = 2,
        momentum = 0.02,
        pretraining_ratio = 0.5,
        verbose = FALSE,
        device = "auto"
        #importance_sample_size = NULL
      )

      super$initialize(
        id = "regr.torch.tabnet",
        packages = "tabnet",
        feature_types = c("logical", "integer", "numeric", "factor", "ordered"),
        param_set = ps,
        properties = c("importance", "missings", "selected_features"),
        man = "mlr3torch::regr.torch.tabnet"
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
