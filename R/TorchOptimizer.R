#' @title Convert to TorchOptimizer
#'
#' @description
#' Converts an object to a [`TorchOptimizer`].
#'
#' @param x (any)\cr
#'   Object to convert to a [`TorchOptimizer`].
#' @param clone (`logical(1)`\cr
#'   Whether to make a deep clone. Default is `FALSE`.
#' @param ... Additional arguments.
#'
#' @return [`TorchOptimizer`]
#' @export
as_torch_optimizer = function(x, clone = FALSE, ...) {
  assert_flag(clone)
  UseMethod("as_torch_optimizer")
}

#' @export
as_torch_optimizer.torch_optimizer_generator = function(x, clone = FALSE) { # nolint
  # clone argument is irrelevant
  TorchOptimizer$new(x, label = deparse(substitute(x))[[1]])
}

#' @export
as_torch_optimizer.TorchOptimizer = function(x, clone = FALSE) { # nolint
  if (clone) x$clone(deep = TRUE) else x
}

#' @export
as_torch_optimizer.character = function(x, clone = FALSE, ...) { # nolint
  # clone argument is irrelevant
  t_opt(x, ...)
}

#' @title Torch Optimizer
#'
#' @usage NULL
#' @name TorchOptimizer
#' @format `r roxy_format(TorchOptimizer)`
#'
#' @description
#' This wraps a `torch::torch_optimizer_generator`.
#' It is commonly used to configure the `optimizer` of a torch learner.
#' Can be used to configure the `optimizer` of a [`ModelDescriptor`].
#'
#' For a list of available optimizers, see [`mlr3torch_optimizers`].
#'
#' @section Construction:
#' `r roxy_construction(TorchOptimizer)`
#'
#' Arguments from [`TorchWrapper`] (except for `generator`) as well as:
#' * `torch_optimizer` :: `torch_optimizer_generator`\cr
#'   The torch optimizer.
#'
#' @section Parameters:
#' Defined by the constructor argument `param_set`.
#'
#' @section Fields:
#' Only fields inherited from [`TorchWrapper`] as well as:
#'
#' @section Methods:
#' Methods inherited from [`TorchWrapper`] as well as:
#' * `generate(params)`\cr
#' (named `list()` of [`nn_parameter`]) -> (`torch_optimizer`)\cr
#' Creates the optimizer for the given parameters.
#'
#' @family torch_wrappers
#' @export
#' @examples
#' # Create a new Torch Optimizer
#' opt = TorchOptimizer$new(optim_adam, label = "adam")
#' # If the param set is not specified, parameters are inferred but are of class ParamUty
#' tochopt$param_set
#'
#' # Create the optimizer for a network
#' net = nn_linear(10, 1)
#' opt = torchopt$generate(net$parameters)
TorchOptimizer = R6::R6Class("TorchOptimizer",
  inherit = TorchWrapper,
  public = list(
    label = NULL,
    optimizer = NULL,
    param_set = NULL,
    packages = NULL,
    initialize = function(torch_optimizer, param_set = NULL,
      id = deparse(substitute(torch_optimizer))[[1]], label = id, packages = NULL) {
     torch_optimizer = assert_class(torch_optimizer, "torch_optimizer_generator") # maybe too strict?
      super$initialize(
        generator = torch_optimizer,
        id = id,
        param_set = param_set,
        packages = packages,
        label = label
      )
    },
    generate = function(params) {
      invoke(self$generator, .args = self$param_set$get_values(), params = params)
    }
  )
)

#' @title Optimizers
#' @description
#' Dictionary of torch optimizers.
#' Use [`t_opt`] for conveniently retrieving optimizers.
#'
#' @section Available Optimizers:
#'
#' * adadelta - [`torch::optim_adadelta`]
#' * adagrad - [`torch::optim_adagrad`]
#' * adam - [`torch::optim_adam`]
#' * asgd - [`torch::optim_asgd`]
#' * lbfgs - [`torch::optim_lbfgs`]
#' * rmsprop - [`torch::optim_rmsprop`]
#' * rprop - [`torch::optim_rprop`]
#' * sgd - [`torch::optim_sgd`]
#'
#' @export
mlr3torch_optimizers = R6Class("DictionaryMlr3torchOptimizers",
  inherit = Dictionary,
  cloneable = FALSE
)$new()


#' @title Optimizers Quick Access
#' @param .key (`character(1)`)\cr
#'   Key of the object to retrieve.
#' @param ... (any)\cr
#'   See description of [`dictionary_sugar_get`].
#' @return A [`TorchOptimizer`]
#' @examples
#' torch_opt = t_opt("adam", lr = 0.1)
#' torch_opt$param_set
#' torch_opt$optimizer
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
      lr           = p_dbl(default = 0.001, lower = 0, upper = Inf, tags = "train"),
      betas        = p_uty(default = c(0.9, 0.999), tags = "train", custom_check = check_betas),
      eps          = p_dbl(default = 1e-08, lower = 1e-16, upper = 1e-4, tags = "train"),
      weight_decay = p_dbl(default = 0, lower = 0, upper = 1, tags = "train"),
      amsgrad      = p_lgl(default = FALSE, tags = "train")
    )
    TorchOptimizer$new(torch::optim_adam, p, "adam")
  }
)


mlr3torch_optimizers$add("sgd",
  function() {
    p = ps(
      lr           = p_dbl(lower = 0, tags = c("required", "train")),
      momentum     = p_dbl(0, 1, default = 0, tags = "train"),
      dampening    = p_dbl(default = 0, lower = 0, upper = 1, tags = "train"),
      weight_decay = p_dbl(0, 1, default = 0, tags = "train"),
      nesterov     = p_lgl(default = FALSE, tags = "train")
    )
    TorchOptimizer$new(torch::optim_sgd, p, "sgd")
  }
)


mlr3torch_optimizers$add("asgd",
  function() {
    p = ps(
      lr           = p_dbl(default = 1e-2, lower = 0, tags = c("required", "train")),
      lambda       = p_dbl(lower = 0, upper = 1, default = 1e-4, tags = "train"),
      alpha        = p_dbl(lower = 0, upper = Inf, default = 0.75, tags = "train"),
      t0           = p_int(lower = 1L, upper = Inf, default = 1e6, tags = "train"),
      weight_decay = p_dbl(default = 0, lower = 0, upper = 1, tags = "train")
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
      lr         = p_dbl(default = 0.01, lower = 0, tags = "train"),
      etas       = p_uty(default = c(0.5, 1.2), tags = "train"),
      step_sizes = p_uty(c(1e-06, 50), tags = "train")
    )
    TorchOptimizer$new(torch::optim_rprop, p, "rprop")
  }
)


mlr3torch_optimizers$add("rmsprop",
  function() {
    p = ps(
      lr           = p_dbl(default = 0.01, lower = 0, tags = "train"),
      alpha        = p_dbl(default = 0.99, lower = 0, upper = 1, tags = "train"),
      eps          = p_dbl(default = 1e-08, lower = 0, upper = Inf, tags = "train"),
      weight_decay = p_dbl(default = 0, lower = 0, upper = 1, tags = "train"),
      momentum     = p_dbl(default = 0, lower = 0, upper = 1, tags = "train"),
      centered     = p_lgl(default = FALSE, tags = "train")
    )
    TorchOptimizer$new(torch::optim_rmsprop, p, "rmsprop")
  }
)


mlr3torch_optimizers$add("adagrad",
  function() {
    p = ps(
      lr                        = p_dbl(default = 0.01, lower = 0, tags = "train"),
      lr_decay                  = p_dbl(default = 0, lower = 0, upper = 1, tags = "train"),
      weight_decay              = p_dbl(default = 0, lower = 0, upper = 1, tags = "train"),
      initial_accumulator_value = p_dbl(default = 0, lower = 0, tags = "train"),
      eps                       = p_dbl(default = 1e-10, lower = 1e-16, upper = 1e-4, tags = "train")
    )
    TorchOptimizer$new(torch::optim_adagrad, p, "adagrad")
  }
)


mlr3torch_optimizers$add("adadelta",
  function() {
    p = ps(
      lr           = p_dbl(default = 1, lower = 0, tags = "train"),
      rho          = p_dbl(default = 0.9, lower = 0, upper = 1, tags = "train"),
      eps          = p_dbl(default = 1e-06, lower = 1e-16, upper = 1e-4, tags = "train"),
      weight_decay = p_dbl(default = 0, lower = 0, upper = 1, tags = "train")
    )
    TorchOptimizer$new(torch::optim_adadelta, p, "adadelta")
  }
)


mlr3torch_optimizers$add("lbfgs",
  function() {
    p = ps(
      lr               = p_dbl(default = 1, lower = 0, tags = "train"),
      max_iter         = p_int(default = 20, lower = 1, tags = "train"),
      max_eval         = p_dbl(default = NULL, lower = 1L, tags = "train", special_vals = list(NULL)),
      tolerance_grad   = p_dbl(default = 1e-07, lower = 0, tags = "train"),
      tolerance_change = p_dbl(default = 1e-09, lower = 0, tags = "train"),
      history_size     = p_int(default = 100L, lower = 1L, tags = "train"),
      line_search_fn   = p_fct(default = NULL, levels = "strong_wolfe", tags = "train", special_vals = list(NULL))
    )
    TorchOptimizer$new(torch::optim_lbfgs, p, "lbfgs")
  }
)
