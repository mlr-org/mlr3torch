
#' @title Convert to TorchOptimizer
#' @export
as_torch_optimizer = function(x, clone = FALSE, ...) {
  assert_flag(clone)
  UseMethod("as_torch_optimizer")
}

#' @export
as_torch_optimizer.torch_optimizer_generator = function(x, clone = FALSE) {
  TorchOptimizer$new(x, label = deparse(substitute(x))[[1]])
}

#' @export
as_torch_optimizer.TorchOptimizer = function(x, clone = FALSE) {
  if (clone) x$clone(deep = TRUE) else x
}

#' @export
as_torch_optimizer.character = function(x, ...) {
  t_opt(x, ...)
}

#' @title TorchOptimizer
#' @export
TorchOptimizer = R6::R6Class("TorchOptimizer",
  public = list(
    label = NULL,
    optimizer = NULL,
    param_set = NULL,
    initialize = function(torch_optimizer, param_set = NULL, label = deparse(substitute(torch_optimizer))[[1]]) {
      assert_r6(param_set, "ParamSet", null.ok = TRUE)
      self$label = assert_string(label)
      self$optimizer = assert_class(torch_optimizer, "torch_optimizer_generator")  # maybe too strict?

      self$param_set = param_set %??% inferps(torch_optimizer, ignore = "params")
    },
    get_optimizer = function(params) {
      invoke(self$optimizer, .args = self$param_set$get_values(), params = params)
    }
  ),
  private = list(
  )
)

#' @title Optimizers
#' @export
mlr3torch_optimizers = R6Class("DictionaryMlr3torchOptimizers",
  inherit = Dictionary,
  cloneable = FALSE
)$new()


#' @title Optimizers Quick Access
#' @export
t_opt = function(.key, ...) {
  dictionary_sugar_get(mlr3torch_optimizers, .key, ...)
}

mlr3torch_optimizers$add("adam",
  function() {
    check_betas = function(x) {
      if (test_numeric(x, lower = 0, upper = 1, any.missing = FALSE, len = 2L)) {
        return(TRUE)
      } else {
        return("Parameter betas invalid, must be a numeric vector of length 2 in (0, 1).")
      }
    }
    p = ps(
      lr = p_dbl(default = 0.001, lower = 0, upper = Inf, tags = c("train", "optimizer")),
      betas = p_uty(default = c(0.9, 0.999), tags = c("train", "optimizer"), custom_check = check_betas),
      eps = p_dbl(default = 1e-08, lower = 1e-16, upper = 1e-4, tags = c("train", "optimizer")),
      weight_decay = p_dbl(default = 0, lower = 0, upper = 1, tags = c("train", "optimizer")),
      amsgrad = p_lgl(default = FALSE, tags = c("train", "optimizer"))
    )
    TorchOptimizer$new(torch::optim_adam, p, "adam")
  }
)


mlr3torch_optimizers$add("sgd",
  function() {
    p = ps(
      lr = p_dbl(default = 0.001, lower = 0, tags = c("required", c("train", "optimizer"))),
      momentum = p_dbl(0, 1, default = 0, tags = c("train", "optimizer")),
      dampening = p_dbl(default = 0, lower = 0, upper = 1, tags = c("train", "optimizer")),
      weight_decay = p_dbl(0, 1, default = 0, tags = c("train", "optimizer")),
      nesterov = p_dbl(default = 0, upper = 1, lower = 0, tags = c("train", "optimizer"))
    )
    TorchOptimizer$new(torch::optim_sgd, p, "sgd")
  }
)


mlr3torch_optimizers$add("asgd",
  function() {
    p = ps(
      lr = p_dbl(default = 1e-2, lower = 0, tags = c("required", c("train", "optimizer"))),
      lambd = p_dbl(lower = 0, upper = 1, default = 1e-4, tags = c("train", "optimizer")),
      alpha = p_dbl(lower = 0, upper = Inf, default = 0.75, tags = c("train", "optimizer")),
      t0 = p_int(lower = 1L, upper = Inf, default = 1e6, tags = c("train", "optimizer")),
      weight_decay = p_dbl(default = 0, lower = 0, upper = 1, tags = c("train", "optimizer"))
    )
    TorchOptimizer$new(torch::optim_asgd, p, "asgd")
  }
)


mlr3torch_optimizers$add("rprop",
  function() {
    check_etas = function(x) {
      if (test_numeric(x, lower = 0, upper = Inf, finite = TRUE, len = 2L)) {
        return(TRUE)
      } else {
        return("Parameter etas invalid, must be a numeric vector of length 2 in (0, Inf).")
      }

    }
    p = ps(
      lr = p_dbl(default = 0.01, lower = 0, tags = c("train", "optimizer")),
      etas = p_uty(default = c(0.5, 1.2), tags = c("train", "optimizer")),
      step_sizes = p_uty(c(1e-06, 50), tags = c("train", "optimizer"))
    )
    TorchOptimizer$new(torch::optim_rprop, p, "rprop")
  }
)


mlr3torch_optimizers$add("rmsprop",
  function() {
    p = ps(
      lr = p_dbl(default = 0.01, lower = 0, tags = c("train", "optimizer")),
      alpha = p_dbl(default = 0.99, lower = 0, upper = 1, tags = c("train", "optimizer")),
      eps = p_dbl(default = 1e-08, lower = 1e-16, upper = 1e-4, tags = c("train", "optimizer")),
      weight_decay = p_dbl(default = 0, lower = 0, upper = 1, tags = c("train", "optimizer")),
      momentum = p_dbl(default = 0, lower = 0, upper = 1, tags = c("train", "optimizer")),
      centered = p_lgl(tags = c("train", "optimizer"))
    )
    TorchOptimizer$new(torch::optim_rmsprop, p, "rmsprop")
  }
)


mlr3torch_optimizers$add("adagrad",
  function() {
    p = ps(
      lr = p_dbl(default = 0.01, lower = 0, tags = c("train", "optimizer")),
      lr_decay = p_dbl(default = 0, lower = 0, upper = 1, tags = c("train", "optimizer")),
      weight_decay = p_dbl(default = 0, lower = 0, upper = 1, tags = c("train", "optimizer")),
      initial_accumulator_value = p_dbl(default = 0, lower = 0, tags = c("train", "optimizer")),
      eps = p_dbl(default = 1e-10, lower = 1e-16, upper = 1e-4, tags = c("train", "optimizer"))
    )
    TorchOptimizer$new(torch::optim_adagrad, p, "adagrad")
  }
)


mlr3torch_optimizers$add("adadelta",
  function() {
    p = ps(
      lr = p_dbl(default = 1, lower = 0, tags = c("train", "optimizer")),
      rho = p_dbl(default = 0.9, lower = 0, upper = 1, tags = c("train", "optimizer")),
      eps = p_dbl(default = 1e-06, lower = 1e-16, upper = 1e-4, tags = c("train", "optimizer")),
      weight_decay = p_dbl(default = 0, lower = 0, upper = 1, tags = c("train", "optimizer"))
    )
    TorchOptimizer$new(torch::optim_adadelta, p, "adadelta")
  }
)


mlr3torch_optimizers$add("lbfgs",
  function() {
    p = ps(
      lr = p_dbl(default = 1, lower = 0, tags = c("train", "optimizer")),
      max_iter = p_int(default = 20, lower = 1, tags = c("train", "optimizer")),
      max_eval = p_dbl(lower = 1L, tags = c("train", "optimizer")),
      tolerance_grad = p_dbl(default = 1e-07, lower = 1e-16, upper = 1e-4, tags = c("train", "optimizer")),
      tolerance_change = p_dbl(default = 1e-09, lower = 1e-16, upper = 1e-4, tags = c("train", "optimizer")),
      history_size = p_int(default = 100L, lower = 1L, tags = c("train", "optimizer")),
      line_search_fn = p_fct(default = "strong_wolfe", levels = c("strong_wolfe"),
        tags = c("train", "optimizer"))
    )
    TorchOptimizer$new(torch::optim_lbfgs, p, "lbfgs")
  }
)


