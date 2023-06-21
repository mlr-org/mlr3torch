#' @title My Little Pony Classification
#'
#' @templateVar id classif.mlp
#' @templateVar param_vals layers = 1, d_hidden = 10
#' @template params_learner
#' @template learner
#' @template learner_example
#'
#' @description
#' Fully connected feed forward network with dropout after each activation function.
#'
#' @section Parameters:
#' Parameters from [`LearnerClassifTorch`], as well as:
#'
#' * `activation` :: `character(1)`\cr
#'   Activation function.
#' * `activation_args` :: named `list()`\cr
#'   A named list with initialization arguments for the activation function.
#' * `layers` :: `integer(1)`\cr
#'   The number of layers.
#' * `d_hidden` :: `numeric(1)`\cr
#'   The dimension of the hidden layers.
#' * `p` :: `numeric(1)`\cr
#'   The dropout probability.
#'
#' @export
LearnerClassifMLP = R6Class("LearnerClassifMLP",
  inherit = LearnerClassifTorch,
  public = list(
    #' @description
    #' Creates a new instance of this [R6][R6::R6Class] class.
    initialize = function(optimizer = t_opt("adam"), loss = t_loss("cross_entropy"), callbacks = list()) {
      param_set = ps(
        activation      = p_fct(default = "relu", tags = "train", levels = mlr_reflections$torch$activations),
        activation_args = p_uty(tags = "train", custom_check = check_list),
        layers          = p_int(lower = 0L, tags = c("train", "required")),
        d_hidden        = p_int(lower = 1L, tags = "train"),
        p               = p_dbl(default = 0.5, lower = 0, upper = 1, tags = "train")
      )
      super$initialize(
        id = "classif.mlp",
        properties = c("twoclass", "multiclass"),
        label = "Multi Layer Perceptron",
        param_set = param_set,
        optimizer = optimizer,
        callbacks = callbacks,
        loss = loss,
        man = "mlr3torch::mlr_learners_classif.mlp",
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


#' @title My Little Pony Regression
#'
#' @templateVar id regr.mlp
#' @templateVar param_vals layers = 1, d_hidden = 10
#' @template params_learner
#' @template learner
#' @template learner_example
#'
#' @inherit mlr_learners_classif.mlp description
#'
#' @section Parameters:
#' Parameters from [`LearnerRegrTorch`], as well as:
#'
#' * `activation` :: `character(1)`\cr
#'   Activation function.
#' * `activation_args` :: named `list()`\cr
#'   A named list with initialization arguments for the activation function.
#' * `layers` :: `integer(1)`\cr
#'   The number of layers.
#' * `d_hidden` :: `numeric(1)`\cr
#'   The dimension of the hidden layers.
#' * `p` :: `numeric(1)`\cr
#'   The dropout probability.
#'
#' @export
LearnerRegrMLP = R6Class("LearnerRegrMLP",
  inherit = LearnerRegrTorch,
  public = list(
    #' @description
    #' Creates a new instance of this [R6][R6::R6Class] class.
    initialize = function(optimizer = t_opt("adam"), loss = t_loss("mse"), callbacks = list()) {
      param_set = ps(
        activation      = p_fct(tags = c("train", "required"), levels = mlr_reflections$torch$activations),
        activation_args = p_uty(tags = c("train", "required"), custom_check = check_list),
        layers          = p_int(lower = 0L, tags = c("train", "required")),
        d_hidden        = p_int(lower = 1L, tags = c("train", "required")),
        p               = p_dbl(lower = 0, upper = 1, tags = c("train", "required"))
      )
      param_set$set_values(
        activation = "relu",
        activation_args = list(),
        p = 0.5
      )
      super$initialize(
        id = "regr.mlp",
        properties = character(),
        label = "Multi Layer Perceptron",
        param_set = param_set,
        optimizer = optimizer,
        callbacks = callbacks,
        loss = loss,
        man = "mlr3torch::mlr_learners_regr.mlp",
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
  activation = activation %??% "relu"
  act = getFromNamespace(paste0("nn_", activation), ns = "torch")
  layers = layers
  d_hidden = d_hidden
  if (layers > 0) assert_true(!is.null(d_hidden))

  out_dim = switch(task_type,
    regr = 1,
    classif = length(task$class_names)
  )
  if (layers == 0L) {
    network = nn_sequential(
      nn_linear(length(task$feature_names), out_dim)
    )
    return(network)
  }

  # This way, dropout_args will have length 0 if p is `NULL`
  dropout_args = list(p = p)
  dropout_args$p = p

  modules = list(
    nn_linear(length(task$feature_names), d_hidden),
    invoke(act, .args = activation_args),
    invoke(nn_dropout, .args = dropout_args)
  )

  for (i in seq_len(layers - 1L)) {
    modules = c(modules, list(
      nn_linear(d_hidden, d_hidden),
      invoke(act, .args = activation_args),
      invoke(nn_dropout, .args = dropout_args)
    ))
  }

  modules = c(modules, list(nn_linear(d_hidden, out_dim)))

  invoke(nn_sequential, .args = modules)
}

register_learner("regr.mlp", LearnerRegrMLP)
register_learner("classif.mlp", LearnerClassifMLP)
