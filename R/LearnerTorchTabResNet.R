#' @title Tabular ResNet
#'
#' @templateVar name tab_resnet
#' @templateVar task_types classif, regr
#' @templateVar param_vals n_blocks = 2, d_block = 10, d_hidden = 20, dropout1 = 0.3, dropout2 = 0.3
#' @template params_learner
#' @template learner
#' @template learner_example
#'
#' @description
#' Tabular resnet.
#'
#' @section Parameters:
#' Parameters from [`LearnerTorch`], as well as:
#' * `n_blocks` :: `integer(1)`\cr
#'   The number of blocks.
#' * `d_block` :: `integer(1)`\cr
#'   The input and output dimension of a block.
#' * `d_hidden` :: `integer(1)`\cr
#'   The latent dimension of a block.
#' * `d_hidden_multiplier` :: `numeric(1)`\cr
#'   Alternative way to specify the latent dimension as `d_block * d_hidden_multiplier`.
#' * `dropout1` :: `numeric(1)`\cr
#'   First dropout ratio.
#' * `dropout2` :: `numeric(1)`\cr
#'    Second dropout ratio.
#' * `shape` :: `integer()` or `NULL`\cr
#'   Shape of the input tensor. Only needs to be provided if the input is a lazy tensor with
#'   unknown shape.
#'
#' @references
#' `r format_bib("gorishniy2021revisiting")`
#' @export
LearnerTorchTabResNet = R6Class("LearnerTorchTabResNet",
  inherit = LearnerTorch,
  public = list(
    #' @description
    #' Creates a new instance of this [R6][R6::R6Class] class.
    initialize = function(task_type, optimizer = NULL, loss = NULL, callbacks = list()) {
      private$.block = PipeOpTorchTabResNetBlock$new()

      check_shape = crate(function(x) check_shape(x, null_ok = TRUE, len = 2L))

      private$.param_set_base =  ps(
        n_blocks = p_int(0, tags = c("train", "required")),
        d_block = p_int(1, tags = c("train", "required")),
        shape = p_uty(tags = "train", custom_check = check_shape)
      )
      param_set = alist(private$.block$param_set, private$.param_set_base)

      super$initialize(
        task_type = task_type,
        id = paste0(task_type, ".tab_resnet"),
        label = "Tabular ResNet",
        param_set = param_set,
        optimizer = optimizer,
        callbacks = callbacks,
        loss = loss,
        man = "mlr3torch::mlr_learners.tab_resnet",
        feature_types = c("numeric", "integer", "lazy_tensor"),
      )
    }
  ),
  private = list(
    .block = NULL,
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
    },
    .network = function(task, param_vals) {
      ingress = if (single_lazy_tensor(task)) {
        po("torch_ingress_ltnsr", id = "num", shape = private$.ingress_tokens(task, param_vals)[[1L]]$shape)
      } else {
        po("torch_ingress_num", id = "num")
      }
      graph = ingress %>>%
        po("nn_linear", out_features = param_vals$d_block) %>>%
        po("nn_block", private$.block, n_blocks = param_vals$n_blocks) %>>%
        po("nn_head")

      md = graph$train(task)[[1L]]
      model_descriptor_to_module(md)
    }
  )
)

PipeOpTorchTabResNetBlock = R6Class("PipeOpTorchTabResNetBlock",
  inherit = PipeOpTorch,
  public = list(
    initialize = function(id = "nn_tab_resnet", param_vals = list()) {
      param_set = ps(
        d_hidden            = p_int(1, default = NULL, tags = "train", special_vals = list(NULL)),
        d_hidden_multiplier = p_dbl(0, default = NULL, tags = "train", special_vals = list(NULL)),
        dropout1            = p_dbl(0, 1, tags = c("train", "required")),
        dropout2            = p_dbl(0, 1, tags = c("train", "required"))
      )
      # otherwise the label method calls $help(), which fails because this pipeop is not
      # exported
      private$.label = "Tabular ResNet Block"
      super$initialize(
        id = id,
        param_set = param_set,
        param_vals = param_vals,
        module_generator = nn_tab_resnet_block
      )
    }
  ),
  private = list(
    .shape_dependent_params = function(shapes_in, param_vals, task) {
      c(param_vals, list(d_block = shapes_in[[1L]][2L]))
    },
    .shapes_out = function(shapes_in, param_vals, task) {
      shapes_in
    }
  )
)

nn_tab_resnet_block = nn_module("nn_tab_resnet_block",
  initialize = function(
    d_block,
    d_hidden = NULL,
    d_hidden_multiplier = NULL,
    dropout1,
    dropout2
  ) {
    assert_int(d_block, lower = 1L)
    if (is.null(d_hidden)) {
      assert_numeric(d_hidden_multiplier, lower = 0, null.ok = TRUE)
      d_hidden = as.integer(d_block * d_hidden_multiplier)
    } else {
      assert_int(d_hidden, lower = 1L)
      assert_true(is.null(d_hidden_multiplier))
    }
    self$normalization = invoke(nn_batch_norm1d, num_features = d_block)
    self$activation = nn_relu()
    self$linear_first = nn_linear(d_block, d_hidden)
    self$dropout_first = nn_dropout(dropout1)
    self$linear_second = nn_linear(d_hidden, d_block)
    self$dropout_second = nn_dropout(dropout2)
  },
  forward = function(input) {
    x = self$normalization(input)
    x = self$linear_first(x)
    x = self$activation(x)
    x = self$dropout_first(x)
    x = self$linear_second(x)
    x = self$dropout_second(x)
    x + input
  }
)

register_learner("regr.tab_resnet", LearnerTorchTabResNet)
register_learner("classif.tab_resnet", LearnerTorchTabResNet)
