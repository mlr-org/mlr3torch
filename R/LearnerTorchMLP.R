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
#' * `p` :: `numeric(1)`\cr
#'   The dropout probability. Is initialized to `0.5`.
#' * `shape` :: `integer()` or `NULL`\cr
#'   The input shape of length 2, e.g. `c(NA, 5)`.
#'   Only needs to be present when there is a lazy tensor input with unknown shape (`NULL`).
#'   Otherwise the input shape is inferred from the number of numeric features.
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
      check_neurons = crate(function(x) check_integerish(x, any.missing = FALSE, lower = 1), .parent = topenv())
      check_shape = crate(function(x) check_shape(x, null_ok = TRUE, len = 2L), .parent = topenv())

      param_set = ps(
        neurons         = p_uty(tags = c("train", "predict"), custom_check = check_neurons),
        p               = p_dbl(lower = 0, upper = 1, tags = "train"),
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
        private$.get_input_shape(task, param_vals$shape)[2L]
      } else {
        length(task$feature_names)
      }
      network = invoke(make_mlp, .args = param_vals, d_in = d_in, d_out = d_out)
      network
    },
    .dataset = function(task, param_vals) {
      if (single_lazy_tensor(task)) {
        param_vals$shape = private$.get_input_shape(task, param_vals$shape)
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
        shape = private$.get_input_shape(task, param_vals$shape)
        assert_shape(shape, len = 2L)
      }
    },
    .get_input_shape = function(s1, s2) {
      if (test_class(s1, "Task")) {
        assert_true(identical(s1$feature_types[, "type"][[1L]], "lazy_tensor"))
        s1 = dd(s1$data(s1$row_roles$use[1L], s1$feature_names)[[1L]])$pointer_shape
      }
      assert_shape(s1, null_ok = TRUE)
      assert_shape(s2, null_ok = TRUE)
      s = unique(discard(list(s1, s2), is.null))
      assert_true(length(s) == 1L)
      s[[1L]]
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
