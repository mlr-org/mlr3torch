#' @title Convert to TorchOptimizer
#'
#' @description
#' Converts an object to a [`TorchOptimizer`].
#'
#' @param x (any)\cr
#'   Object to convert to a [`TorchOptimizer`].
#' @param clone (`logical(1)`\cr
#'   Whether to make a deep clone. Default is `FALSE`.
#' @param ... (any)\cr
#'   Additional arguments.
#'   Currently used to pass additional constructor arguments to [`TorchOptimizer`] for objects of type
#'   `torch_optimizer_generator`.
#'
#' @family Torch Wrapper
#'
#' @return [`TorchOptimizer`]
#' @export
as_torch_optimizer = function(x, clone = FALSE, ...) {
  assert_flag(clone)
  UseMethod("as_torch_optimizer")
}

#' @export
as_torch_optimizer.torch_optimizer_generator = function(x, clone = FALSE, id = deparse(substitute(x))[[1L]], ...) { # nolint
  # clone argument is irrelevant
  TorchOptimizer$new(x, id = id, ...)
}

#' @export
as_torch_optimizer.TorchOptimizer = function(x, clone = FALSE, ...) { # nolint
  if (clone) x$clone(deep = TRUE) else x
}

#' @export
as_torch_optimizer.character = function(x, clone = FALSE, ...) { # nolint
  # clone argument is irrelevant
  t_opt(x, ...)
}

#' @title Torch Optimizer
#'
#' @name TorchOptimizer
#'
#' @description
#' This wraps a `torch::torch_optimizer_generator`.
#' It is commonly used to configure the `optimizer` of a torch learner.
#' Can be used to configure the `optimizer` of a [`ModelDescriptor`].
#'
#' For a list of available optimizers, see [`mlr3torch_optimizers`].
#'
#' @section Parameters:
#' Defined by the constructor argument `param_set`.
#' @family Torch Wrapper
#' @export
#' @examples
#' # Create a new Torch Optimizer
#' torchopt = TorchOptimizer$new(optim_adam, label = "adam")
#' # If the param set is not specified, parameters are inferred but are of class ParamUty
#' torchopt$param_set
#'
#' # Create the optimizer for a network
#' net = nn_linear(10, 1)
#' opt = torchopt$generate(net$parameters)
TorchOptimizer = R6::R6Class("TorchOptimizer",
  inherit = TorchWrapper,
  public = list(
    #' @description
    #' Creates a new instance of this [R6][R6::R6Class] class.
    #' @param torch_optimizer (`torch_optimizer_generator`)\cr
    #'   The torch optimizer.
    #' @template param_param_set
    #' @template param_id
    #' @template param_label
    #' @template param_packages
    #' @template param_man
    initialize = function(torch_optimizer, param_set = NULL, id = deparse(substitute(torch_optimizer))[[1L]],
      label = capitalize(id), packages = NULL, man = NULL) {
      torch_optimizer = assert_class(torch_optimizer, "torch_optimizer_generator") # maybe too strict?
      if (test_r6(param_set, "ParamSet")) {
        if ("params" %in% param_set$ids()) {
          stopf("The name 'params' is reserved for the network parameters.")
        }
      } else {
        param_set = inferps(torch_optimizer, ignore = "params")
      }
      super$initialize(
        generator = torch_optimizer,
        id = id,
        param_set = param_set,
        packages = packages,
        label = label,
        man = man
      )
    },
    #' @description
    #' Instantiates the optimizer.
    #' @param params The `$parameters` of the network.
    generate = function(params) {
      require_namespaces(self$packages)
      invoke(self$generator, .args = self$param_set$get_values(), params = params)
    }
  )
)

#' @title Optimizers
#'
#' @description
#' Dictionary of torch optimizers.
#' Use [`t_opt`] for conveniently retrieving optimizers.
#' Can be converted to a [`data.table`] using `as.data.table`.
#'
#' @section Available Optimizers:
#' `r paste0(mlr3torch_optimizers$keys(), collapse = ", ")`
#'
#' @family Torch Wrapper
#' @family Dictionary
#' @export
#' @examples
#' mlr3torch_optimizers$get("adam")
#' # is equivalent to
#' t_opt("adam")
#' # convert to a data.table
#' as.data.table(mlr3torch_optimizers)
mlr3torch_optimizers = R6Class("DictionaryMlr3torchOptimizers",
  inherit = Dictionary,
  cloneable = FALSE
)$new()

#' @export
as.data.table.DictionaryMlr3torchOptimizers = function(x, ...) {
  setkey(map_dtr(x$keys(), function(key) {
    opt = x$get(key)
    list(
      key = key,
      label = opt$label,
      packages = paste0(opt$packages, collapse = ",")
    )}), "key")[]
}

#' @title Optimizers Quick Access
#' @param .key (`character(1)`)\cr
#'   Key of the object to retrieve.
#' @param ... (any)\cr
#'   See description of [`dictionary_sugar_get`].
#' @return A [`TorchOptimizer`]
#' @export
#' @family Torch Wrapper
#' @family Dictionary
#' @examples
#' t_opt("adam", lr = 0.1)
#' # get the dictionary
#' t_opt()
t_opt = function(.key, ...) {
  UseMethod("t_opt")
}

#' @export
t_opt.character = function(.key, ...) { #nolint
  dictionary_sugar_inc_get(mlr3torch_optimizers, .key, ...)
}

#' @export
t_opt.NULL = function(.key, ...) { # nolint
  # class is NULL if .key is missing
  dictionary_sugar_get(mlr3torch_optimizers)
}

#' @rdname t_opt
#' @param .keys (`character()`)\cr
#'   The keys of the optimizers.
#' @export
#' @examples
#' t_opts(c("adam", "sgd"))
#' # get the dictionary
#' t_opts()
t_opts = function(.keys, ...) {
  UseMethod("t_opts")
}


#' @export
t_opts.character = function(.keys, ...) { # nolint
  dictionary_sugar_inc_mget(mlr3torch_optimizers, .keys, ...)
}

#' @export
t_opts.NULL = function(.keys, ...) { # nolint
  # class is NULL if .keys is missing
  dictionary_sugar_mget(mlr3torch_optimizers)
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
    TorchOptimizer$new(
      torch_optimizer = torch::optim_adam,
      param_set = p,
      id = "adam",
      label = "Adaptive Moment Estimation",
      man = "torch::optim_adam"
    )
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
    TorchOptimizer$new(
      torch_optimizer = torch::optim_sgd,
      param_set = p,
      id = "sgd",
      label = "Stochastic Gradient Descent",
      man = "torch::optim_sgd"
    )
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
    TorchOptimizer$new(
      torch_optimizer = torch::optim_asgd,
      param_set = p,
      id = "asgd",
      label = "SGD with Adaptive Batch Size",
      man = "torch::optim_asgd"
    )
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
    TorchOptimizer$new(
      torch_optimizer = torch::optim_rprop,
      param_set = p,
      id = "rprop",
      label = "Resilient Backpropagation",
      man = "torch::optim_rprop"
    )
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
    TorchOptimizer$new(
      torch_optimizer = torch::optim_rmsprop,
      param_set = p,
      id = "rmsprop",
      label = "Root Mean Square Propagation",
      man = "torch::optim_rmsprop"
    )
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
    TorchOptimizer$new(
      torch_optimizer = torch::optim_adagrad,
      param_set = p,
      id = "adagrad",
      label = "Adaptive Gradient algorithm",
      man = "torch::optim_adagrad"
    )
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
    TorchOptimizer$new(torch_optimizer = torch::optim_adadelta,
      param_set = p,
      id = "adadelta",
      label = "Adaptive Learning Rate Method``",
      man = "torch::optim_adadelta"
    )
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
    TorchOptimizer$new(
      torch_optimizer = torch::optim_lbfgs,
      param_set = p,
      id = "lbfgs",
      label = "Limited-memory BFGS",
      man = "torch::optim_lbfgs"
    )
  }
)
