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
#' * `r roxy_param_loss`
#'
#' @section State: See [`LearnerClassifTorchAbstract`].
#'
#' @section Parameters:
#' Parameters from [`LearnerClassifTorchAbstract`], as well as:
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
#' @section Fields: `r roxy_fields(LearnerClassifMLP)`
#' @section Methods: `r roxy_methods(LearnerClassifMLP)`
#'
#' @section Internals:
#' A [`nn_sequential()`] is generated for the given parameter values.
#'
#'
#' @family Learner
#' @export
LearnerClassifMLP = R6Class("LearnerClassifMLP",
  inherit = LearnerClassifTorchAbstract,
  public = list(
    initialize = function(optimizer = t_opt("adam"), loss = t_loss("cross_entropy")) {
      param_set = ps(
        activation      = p_fct(default = "relu", tags = "train", levels = mlr3torch_activations),
        activation_args = p_uty(tags = "train", custom_check = check_list),
        layers          = p_int(lower = 0L, tags = c("train", "required")),
        d_hidden        = p_int(lower = 1L, tags = c("train", "required")),
        p               = p_dbl(default = 0.5, lower = 0, upper = 1, tags = "train")
      )
      param_set$values = list(activation = "relu")
      super$initialize(
        id = "classif.mlp",
        properties = c("twoclass", "multiclass", "hotstart_forward"),
        label = "Multi Layer Perceptron",
        param_set = param_set,
        optimizer = optimizer,
        loss = loss,
        man = "mlr3torch::mlr_learners_classif.mlp",
        feature_types = c("numeric", "integer")
      )
    }
  ),
  private = list(
    .network = function(task, param_vals) {
      act = getFromNamespace(paste0("nn_", param_vals$activation), ns = "torch")
      
      d_hidden = param_vals$d_hidden
      layers = param_vals$layers
      if (layers == 0L) {
        network = nn_sequential(
          nn_linear(length(task$feature_names), length(task$class_names))
        )
        return(network)
      }
      dropout_args = if (is.null(param_vals$p)) list() else list(p = param_vals$p)

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
      ingress = TorchIngressToken(task$feature_names, batchgetter_num, c(NA, length(task$feature_names)))

      task_dataset(
        task,
        feature_ingress_tokens = list(input = ingress),
        target_batchgetter = crate(function(data, device) {
          torch_tensor(data = as.integer(data[[1L]]), dtype = torch_long(), device)
        })
      )
    }
  )
)

register_learner("classif.mlp", LearnerClassifMLP)
