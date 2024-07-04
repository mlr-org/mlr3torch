#' @title Loss Configuration
#'
#' @name mlr_pipeops_torch_loss
#'
#' @description
#' Configures the loss of a deep learning model.
#'
#' @template pipeop_torch_channels_default
#' @template pipeop_torch_state_default
#'
#' @section Parameters:
#' The parameters are defined dynamically from the loss set during construction.
#' @section Internals:
#' During training the loss is cloned and added to the [`ModelDescriptor`].
#'
#' @family PipeOps
#' @family Model Configuration
#'
#' @export
#' @examplesIf torch::torch_is_installed()
#' po_loss = po("torch_loss", loss = t_loss("cross_entropy"))
#' po_loss$param_set
#' mdin = po("torch_ingress_num")$train(list(tsk("iris")))
#' mdin[[1L]]$loss
#' mdout = po_loss$train(mdin)[[1L]]
#' mdout$loss
PipeOpTorchLoss = R6Class("PipeOpTorchLoss",
  inherit = PipeOp,
  public = list(
    #' @description
    #' Creates a new instance of this [R6][R6::R6Class] class.
    #' @param loss ([`TorchLoss`] or `character(1)` or `nn_loss`)\cr
    #'   The loss (or something convertible via [`as_torch_loss()`]).
    #' @template params_pipelines
    initialize = function(loss, id = "torch_loss", param_vals = list()) {
      private$.loss = as_torch_loss(loss, clone = TRUE)
      input = data.table(name = "input", train = "ModelDescriptor", predict = "Task")
      output = data.table(name = "output", train = "ModelDescriptor", predict = "Task")
      super$initialize(
        id = id,
        param_set = alist(private$.loss$param_set),
        param_vals = param_vals,
        input = input,
        output = output,
        packages = private$.loss$packages
      )
    }
  ),
  private = list(
    .train = function(inputs) {
      if (!test_null(inputs[[1L]]$loss)) {
        stopf("The loss of the model descriptor is already configured.")
      }
      assert_true(is.null(inputs[[1L]]$loss))
      inputs[[1]]$loss = private$.loss$clone(deep = TRUE)
      self$state = list()
      inputs
    },
    .predict = function(inputs) {
      inputs
    },
    .loss = NULL,
    .additional_phash_input = function() self$loss$phash
  )
)

# We set an arbitrary loss, so Dict -> DT conversion works

#' @include zzz.R TorchLoss.R
register_po("torch_loss", PipeOpTorchLoss, metainf = list(loss = t_loss("mse")))
