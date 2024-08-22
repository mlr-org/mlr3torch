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
#' * `d_hidden_multiplier` :: `integer(1)`\cr
#'   Alternative way to specify the latent dimension as `d_block * d_hidden_multiplier`.
#' * `dropout1` :: `numeric(1)`\cr
#'   First dropout ratio.
#' * `dropout2` :: `numeric(1)`\cr
#'    Second dropout ratio.
#'
#' @references
#' `r format_bib("gorishniy2021revisiting")`
#'
#' @export
LearnerTorchTabResNet = R6Class("LearnerTorchTabResNet",
  inherit = LearnerTorch,
  public = list(
    #' @description
    #' Creates a new instance of this [R6][R6::R6Class] class.
    initialize = function(task_type, optimizer = NULL, loss = NULL, callbacks = list()) {
      private$.block = PipeOpTorchTabResNetBlock$new()

      properties = switch(task_type,
        regr = character(0),
        classif = c("twoclass", "multiclass")
      )

      private$.param_set_base =  ps(
        n_blocks = p_int(1, tags = c("train", "required")),
        d_block = p_int(1, tags = c("train", "required"))
      )
      param_set = alist(private$.block$param_set, private$.param_set_base)

      super$initialize(
        task_type = task_type,
        id = paste0(task_type, ".tab_resnet"),
        properties = properties,
        label = "Tabular ResNet",
        param_set = param_set,
        optimizer = optimizer,
        callbacks = callbacks,
        loss = loss,
        man = "mlr3torch::mlr_learners.tab_resnet",
        feature_types = c("numeric", "integer"),
      )
    }
  ),
  private = list(
    .block = NULL,
    .dataset = function(task, param_vals) {
      dataset_num(task, param_vals)
    },
    .network = function(task, param_vals) {
      graph = po("torch_ingress_num") %>>%
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
        d_hidden_multiplier = p_int(1, default = NULL, tags = "train", special_vals = list(NULL)),
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
      assert_int(d_hidden_multiplier, lower = 1L)
      d_hidden = d_block * d_hidden_multiplier
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
