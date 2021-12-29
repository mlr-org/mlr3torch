LearnerRegrNet = R6::R6Class("LearnerRegrNet",
  inherit = LearnerRegr,

  public = list(
    initialize = function(architecture, optimizer, loss, callback) {
      super$initilize(
        id = "regr.torch.net",
        packages = c("torch"),
        feature_type = c("integer", "integer", "numneric", "factor", "ordered"),
        predict_types = c("response"),
        param_set = ParamSet$new(),
        properties = c(),
        man = "mlr3torch::mlr_learners_regr.torch.net"
      )
    }
  ),
  private = list(
    .train_once = function(task, i) {

    }
  )
)
