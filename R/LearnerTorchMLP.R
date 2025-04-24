#' @title Multi Layer Perceptron
#'
#' @templateVar name mlp
#' @templateVar task_types classif, regr
#' @templateVar param_vals neurons = 10
#' @template params_learner
#' @template learner
#' @template learner_example
#'
#' @description
#' Fully connected feed forward network with dropout after each activation function.
#' The features can either be a single [`lazy_tensor`] or one or more numeric columns (but not both).
#'
#' @section Parameters:
#' Parameters from [`LearnerTorch`], as well as:
#'
#' * `activation` :: `[nn_module]`\cr
#'   The activation function. Is initialized to [`nn_relu`][torch::nn_relu].
#' * `activation_args` :: named `list()`\cr
#'   A named list with initialization arguments for the activation function.
#'   This is intialized to an empty list.
#' * `neurons` :: `integer()`\cr
#'   The number of neurons per hidden layer. By default there is no hidden layer.
#'   Setting this to `c(10, 20)` would have a the first hidden layer with 10 neurons and the second with 20.
#' * `n_layers` :: `integer()`\cr
#'   The number of layers. This parameter must only be set when `neurons` has length 1.
#' * `p` :: `numeric(1)`\cr
#'   The dropout probability. Is initialized to `0.5`.
#' * `shape` :: `integer()` or `NULL`\cr
#'   The input shape of length 2, e.g. `c(NA, 5)`.
#'   Only needs to be present when there is a lazy tensor input with unknown shape (`NULL`).
#'   Otherwise the input shape is inferred from the number of numeric features.
#'
#' @references
#' `r format_bib("gorishniy2021revisiting")`
#'
#' @export
LearnerTorchMLP = R6Class("LearnerTorchMLP",
  inherit = LearnerTorch,
  public = list(
    #' @description
    #' Creates a new instance of this [R6][R6::R6Class] class.
    initialize = function(task_type, optimizer = NULL, loss = NULL, callbacks = list()) {
      check_activation = crate(function(x) check_class(x, "nn_module"))
      check_activation_args = crate(function(x) check_list(x, names = "unique"))
      check_neurons = crate(function(x) check_integerish(x, any.missing = FALSE, lower = 1))
      check_shape = crate(function(x) check_shape(x, null_ok = TRUE, len = 2L))

      param_set = ps(
        neurons         = p_uty(tags = c("train", "predict"), custom_check = check_neurons),
        p               = p_dbl(lower = 0, upper = 1, tags = "train"),
        n_layers        = p_int(lower = 1L, tags = "train"),
        activation      = p_uty(tags = c("required", "train"), custom_check = check_nn_module),
        activation_args = p_uty(tags = c("required", "train"), custom_check = check_activation_args),
        shape           = p_uty(tags = "train", custom_check = check_shape)
      )

      param_set$set_values(
        activation = nn_relu,
        activation_args = list(),
        neurons = integer(0),
        p = 0.5
      )

      super$initialize(
        task_type = task_type,
        id = paste0(task_type, ".mlp"),
        label = "Multi Layer Perceptron",
        param_set = param_set,
        optimizer = optimizer,
        callbacks = callbacks,
        loss = loss,
        man = "mlr3torch::mlr_learners.mlp",
        feature_types = c("numeric", "integer", "lazy_tensor")
      )
    }
  ),
  private = list(
    .network = function(task, param_vals) {
      d_out = output_dim_for(task)
      d_in = private$.ingress_tokens(task, param_vals)[[1L]]$shape[2L]
      network = invoke(make_mlp, .args = param_vals, d_in = d_in, d_out = d_out)
      network
    },
    .ingress_tokens = function(task, param_vals) {
      token = if (single_lazy_tensor(task)) {
        shape = param_vals$shape %??% lazy_shape(task$head(1L)[[task$feature_names]])
        if (is.null(shape)) {
          stopf("Learner '%s' received task '%s' with lazy tensor feature '%s' with unknown shape. Please specify the learner's `shape` parameter.", self$id, task$id, task$feature_names) # nolint
        } else if (is.null(param_vals$shape)) {
          msg = check_shape(shape, len = 2L)
          if (!isTRUE(msg)) {
            stopf("Learner '%s' received task '%s' with lazy_tensor column of shape '%s', but the learner expects an input shape of length 2.", self$id, task$id, shape_to_str(shape))
          }
        }
        ingress_ltnsr(shape = shape)
      } else {
        ingress_num(shape = c(NA, length(task$feature_names)))
      }
      list(input = token)
    }
  )
)


# shape is (NA, x) if preesnt
make_mlp = function(task, d_in, d_out, activation, neurons = integer(0), p, activation_args, n_layers = NULL, ...) {
  # This way, dropout_args will have length 0 if p is `NULL`

  if (!is.null(n_layers)) {
    if (length(neurons) != 1L) {
      stopf("Can only supply `n_layers` when neurons has length 1.")
    }
    neurons = rep(neurons, n_layers)
  }

  dropout_args = list()
  dropout_args$p = p
  prev_dim = d_in
  modules = list()
  for (n in neurons) {
    modules = append(modules, list(
      nn_linear(
        in_features = prev_dim,
        out_features = n),
      invoke(activation, .args = activation_args),
      invoke(nn_dropout, .args = dropout_args)
    ))
    prev_dim = n
  }
  modules = c(modules, list(nn_linear(prev_dim, d_out)))
  invoke(nn_sequential, .args = modules)
}

register_learner("regr.mlp", LearnerTorchMLP)
register_learner("classif.mlp", LearnerTorchMLP)
