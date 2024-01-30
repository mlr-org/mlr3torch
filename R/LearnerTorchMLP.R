#' @title My Little Pony
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
#' The features can either be a single [`lazy_tensor`] or one or more numeric columns.
#'
#' @section Parameters:
#' Parameters from [`LearnerTorch`], as well as:
#'
#' * `activation` :: `[nn_module]`\cr
#'   The activation function. Is initialized to [`nn_relu`].
#' * `activation_args` :: named `list()`\cr
#'   A named list with initialization arguments for the activation function.
#'   This is intialized to an empty list.
#' * `neurons` :: `integer()`\cr
#'   The number of neurons per hidden layer.
#'   By default there is no hidden layer.
#' * `p` :: `numeric(1)`\cr
#'   The dropout probability.
#'   Is initialized to `0.5`.
#' * `shape` :: `integer()` or `NULL`\cr
#'   The input shape.
#'   Only needs to be present specified when there is a lazy tensor input with unknown shape.
#'
#' @export
LearnerTorchMLP = R6Class("LearnerTorchMLP",
  inherit = LearnerTorch,
  public = list(
    #' @description
    #' Creates a new instance of this [R6][R6::R6Class] class.
    initialize = function(task_type, optimizer = NULL, loss = NULL, callbacks = list()) {
      check_activation = crate(function(x) check_class(x, "nn_module"), .parent = topenv())
      check_activation_args = crate(function(x) check_list(x, names = "unique"), .parent = topenv())
      param_set = ps(
        neurons         = p_uty(default = integer(0), tags = "train", custom_check = crate(function(x) {
          check_integerish(x, any.missing = FALSE, lower = 1)
        })),
        p               = p_dbl(lower = 0, upper = 1, tags = c("required", "train")),
        activation      = p_uty(tags = c("required", "train"), custom_check = check_activation),
        activation_args = p_uty(tags = c("required", "train"), custom_check = check_activation_args),
        shape           = p_uty(default = NULL, tags = "train", custom_check = crate(function(x) {
            check_shape(x, null_ok = TRUE)
        }, .parent = topenv()))
      )
      param_set$set_values(
        activation = nn_relu,
        p = 0.5,
        activation_args = list()
      )
      properties = switch(task_type,
        regr = character(0),
        classif = c("twoclass", "multiclass")
      )

      super$initialize(
        task_type = task_type,
        id = paste0(task_type, ".mlp"),
        properties = properties,
        label = "My Little Powny",
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
      # verify_train_task was already called beforehand, so we can make some assumptions
      d_out = get_nout(task)
      d_in = if (single_lazy_tensor(task)) {
        get_unique_shape(task, param_vals$shape)[2L]
      } else {
        length(task$feature_names)
      }
      network = invoke(make_mlp, .args = param_vals, d_in = d_in, d_out = d_out)
      network
    },
    .dataset = function(task, param_vals) {
      if (single_lazy_tensor(task)) {
        param_vals$shape = get_unique_shape(task, param_vals$shape)
        dataset_ltnsr(task, param_vals)
      } else {
        dataset_num(task, param_vals)
      }
    },
    .verify_train_task = function(task, param_vals) {
      features = task$feature_types[, "type"][[1L]]
      lazy_tensor_input = identical(features, "lazy_tensor")
      assert(check_true(lazy_tensor_input), check_false(some(features, function(x) x == "lazy_tensor")))

      if (lazy_tensor_input) {
        shape = get_unique_shape(task, param_vals$shape)
        assert_shape(shape, len = 2L)
      }
    }
  )
)

single_lazy_tensor = function(task) {
  identical(task$feature_types[, "type"][[1L]], "lazy_tensor")
}

# shape is (NA, x) if preesnt
make_mlp = function(task, d_in, d_out, activation, neurons = integer(0), p, activation_args, ...) {
  # This way, dropout_args will have length 0 if p is `NULL`
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
