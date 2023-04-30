#' @title Multi Layer Perceptron
#'
#' @usage NULL
#' @name mlr_learners_classif.mlp
#' @format `r roxy_format(LearnerClassifMLP)`
#'
#' @description
#' Simple multi layer perceptron with dropout.
#'
#' @section Construction: `r roxy_construction(LearnerClassifMLP)`
#'
#' @section State: See [`LearnerClassifTorch`].
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
#' @section Fields: `r roxy_fields_inherit(LearnerClassifMLP)`
#' @section Methods: `r roxy_methods_inherit(LearnerClassifMLP)`
#'
#' @section Internals:
#' A [`nn_sequential()`] is generated for the given parameter values.
#'
#' @family Learner
#' @export
LearnerClassifMLP = R6Class("LearnerClassifMLP",
  inherit = LearnerClassifTorch,
  public = list(
    initialize = function(optimizer = t_opt("adam"), loss = t_loss("cross_entropy"), callbacks = list()) {
      param_set = ps(
        activation      = p_fct(default = "relu", tags = "train", levels = mlr3torch_activations),
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

      activation = param_vals$activation %??% "relu"
      act = getFromNamespace(paste0("nn_", activation), ns = "torch")
      layers = param_vals$layers
      d_hidden = param_vals$d_hidden
      if (layers > 0) assert_true(!is.null(d_hidden))

      if (layers == 0L) {
        network = nn_sequential(
          nn_linear(length(task$feature_names), length(task$class_names))
        )
        return(network)
      }

      # This way, dropout_args will have length 0 if p is `NULL`
      dropout_args = list(p = param_vals$p)
      dropout_args$p = param_vals$p

      modules = list(
        nn_linear(length(task$feature_names), d_hidden),
        invoke(act, .args = param_vals$activation_args),
        invoke(nn_dropout, .args = dropout_args)
      )

      for (i in seq_len(layers - 1L)) {
        modules = c(modules, list(
          nn_linear(d_hidden, d_hidden),
          invoke(act, .args = param_vals$activation_args),
          invoke(nn_dropout, .args = dropout_args)
        ))
      }

      modules = c(modules, list(nn_linear(d_hidden, length(task$class_names))))

      invoke(nn_sequential, .args = modules)
    },
    .dataset = function(task, param_vals) {
      dataset_num(self, task, param_vals)
    }
  )
)

register_learner("classif.mlp", LearnerClassifMLP)
