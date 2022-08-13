
#' @title Convert to TorchLoss
#' @export
as_paramd_loss = function(x, clone = FALSE) {
  assert_flag(clone)
  UseMethod("as_paramd_loss")
}

#' @export
as_paramd_loss.nn_loss = function(x, clone = FALSE) {
  TorchLoss$new(x, label = deparse(substitute(x))[[1]])
}

#' @export
as_paramd_loss.TorchLoss = function(x, clone = FALSE) {
  if (clone) x$clone(deep = TRUE) else x
}

#' @title TorchLoss
#' @export
TorchLoss = R6::R6Class("TorchLoss",
  public = list(
    label = NULL,
    tasktypes = NULL,
    loss = NULL,
    param_set = NULL,
    initialize = function(torch_loss, tasktypes = NULL, param_set = NULL, label = deparse(substitute(torch_loss))[[1]]) {
      assert_r6(param_set, "ParamSet", null.ok = TRUE)
      self$tasktypes = assert_subset(tasktypes, mlr_reflections$task_types$type)
      self$label = assert_string(label)
      self$loss = assert_class(torch_loss, "nn_loss")  # maybe too strict?

      self$param_set = param_set %??% inferps(torch_loss)
    },
    get_loss = function() {
      invoke(self$loss, .args = self$param_set$get_values())
    }
  ),
  private = list(
  )
)

#' @title Losses
#' @export
mlr3torch_losses = R6Class("DictionaryMlr3torchLosses",
  inherit = Dictionary,
  cloneable = FALSE
)$new()


#' @title Losses Quick Access
#' @export
t_loss = function(.key, ...) {
  dictionary_sugar_get(mlr3torch_losses, .key, ...)
}

mlr3torch_losses$add("mse", function() {
    p = ps(reduction = p_fct(levels = c("mean", "sum"), default = "mean", tags = "train"))
    TorchLoss$new(torch::nn_mse_loss, "regr", p, "mse")
  }
)


mlr3torch_losses$add("l1", function() {
    p = ps()
    TorchLoss$new(torch::nn_l1_loss, "regr", p, "l1")
  }
)


mlr3torch_losses$add("cross_entropy", function() {
    p = ps(
      weight = p_uty(),
      ignore_index = p_int(),
      reduction = p_fct(levels = c("mean", "sum"), default = "mean")
    )
    TorchLoss$new(torch::nn_cross_entropy_loss, "classif", p, "cross_entropy")
  }
)
