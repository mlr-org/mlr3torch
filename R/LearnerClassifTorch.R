#'
LearnerClassifTorch = R6Class("LearnerClassifTorch",
  inherit = LearnerClassifTorchAbstract,
  public = list(
    #' @description Initializes an object of class LearnerClassifTorch
    #' @param .architecture (mlr3torch::Architecture || torch::nn_module)
    initialize = function(id = "classif.torch", param_vals = list(), .architecture,
      .optimizer, .criterion) {
      param_set = ps(
        architecture = p_uty()
      )
      feature_types = architecture$feature_types

      super$initialize(
        id = id,
        feature_types = feature_types,
        predict_types = "response",
        .optimizer = .optimizer,
        properties = c(),
        packages = "torch",
        task_type = "classif"
      )
    }
  ),
  private = list(
    .train = function(task) {
      pars = self$param_set$get_values(tag = "train")

      if (!length(self$state)) {
        private$.build(pars)
      }
      pars = self$param_set$get_values(tag = "train")
    },
    .build = function(pars) {
      if (inherits(pars[["architecture"]], "Architecture")) {
        model = reduce_architecture(pars[["architecture"]], task)[["model"]]
      } else {
        model = pars[["architecture"]]
      }
      list(
        model = model,
        optimizer = mlr3misc::invoke(pars[["optimizer"]], .args = pars[["optimizer_args"]],
          params = model$parameters
        ),
        criterion = mlr3misc::invoke(pars[["criterion"]], .args = pars[["criterion_args"]])
      )
    },
    .init_optimizer = function() {

    }
  )

)

LearnerClassifTorch = R6Class("LearnerClassifTorch",
  inherit = LearnerTorch,
  public = list(
    initialize = function(id = "classif.t") {
      # TODO: modify input checks to classification
      super$initialize(
        task_type = "classif",
        predict_types = c("response", "prob"),
        param_set = dl_paramset(),
        properties = c("twoclass", "multiclass")
      )
    },
    .predict = function(task) {
      pred_mat = super$.predict(task)
      response = torch_max(pred_mat, dim = 2L)[[2L]]$to(device = "cpu")
      response = recover_factor(response)
      if (self$predict_type == "response") {
        return(list(response = response))
      }
      if (self$predict_type == "prob") {
        pred_matrix = as.matrix(pred_mat$to(device = "cpu"))
        colnames(pred_matrix = levels)
        return(list(prob = pred_matrix, response = response))
      }
    }
  )
)

#' Recover the factor encoding for the prediction
recover_factor = function(target, task) {
  levels = levels(task$data(cols = task$col_roles$target))
  target_fct = structure(targe, class = "factor", levels = levels)

}
