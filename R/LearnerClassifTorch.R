LearnerClassifTorch = R6Class("LearnerClassifTorch",
  inherit = LearnerTorch,
  public = list(
    initialize = function() {
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
