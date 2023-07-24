#' @title My Little Pony
#'
#' @templateVar name mlp
#' @templateVar task_types classif, regr
#' @templateVar param_vals layers = 1, d_hidden = 10
#' @template params_learner
#' @template learner
#' @template learner_example
#'
#' @description
#' Fully connected feed forward network with dropout after each activation function.
#'
#' @section Parameters:
#' Parameters from [`LearnerTorch`], as well as:
#'
#' * `activation` :: `[nn_module]`\cr
#'   The activation function. Is initialized to [`nn_relu`].
#' * `activation_args` :: named `list()`\cr
#'   A named list with initialization arguments for the activation function.
#'   This is intialized to an empty list.
#' * `layers` :: `integer(1)`\cr
#'   The number of layers.
#' * `d_hidden` :: `numeric(1)`\cr
#'   The dimension of the hidden layers.
#' * `p` :: `numeric(1)`\cr
#'   The dropout probability.
#'   Is initialized to `0.5`.
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
        layers          = p_int(lower = 0L, tags = c("required", "train")),
        d_hidden        = p_int(lower = 1L, tags = c("required", "train")),
        p               = p_dbl(lower = 0, upper = 1, tags = c("required", "train")),
        activation      = p_uty(tags = c("required", "train"), custom_check = check_activation),
        activation_args = p_uty(tags = c("required", "train"), custom_check = check_activation_args)
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
        feature_types = c("numeric", "integer")
      )
    }
  ),
  private = list(
    .network = function(task, param_vals) {
      make_mlp(
        task = task,
        activation = param_vals$activation,
        layers = param_vals$layers,
        d_hidden = param_vals$d_hidden,
        p = param_vals$p,
        activation_args = param_vals$activation_args
      )
    },
    .dataset = function(task, param_vals) {
      dataset_num(task, param_vals)
    }
  )
)

make_mlp = function(task, activation, layers, d_hidden, p, activation_args) {
  task_type = task$task_type
  layers = layers
  d_hidden = d_hidden
  if (layers > 0) assert_true(!is.null(d_hidden))

  out_dim = get_nout(task)
  if (layers == 0L) {
    network = nn_sequential(
      nn_linear(length(task$feature_names), out_dim)
    )
    return(network)
  }

  # This way, dropout_args will have length 0 if p is `NULL`
  dropout_args = list()
  dropout_args$p = p

  modules = list(
    nn_linear(length(task$feature_names), d_hidden),
    invoke(activation, .args = activation_args),
    invoke(nn_dropout, .args = dropout_args)
  )

  for (i in seq_len(layers - 1L)) {
    modules = c(modules, list(
      nn_linear(d_hidden, d_hidden),
      invoke(activation, .args = activation_args),
      invoke(nn_dropout, .args = dropout_args)
    ))
  }

  modules = c(modules, list(nn_linear(d_hidden, out_dim)))

  invoke(nn_sequential, .args = modules)
}

register_learner("regr.mlp", LearnerTorchMLP)
register_learner("classif.mlp", LearnerTorchMLP)