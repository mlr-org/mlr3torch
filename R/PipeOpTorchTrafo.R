#' @title Base Class for Lazy Transformations
#' @name mlr_pipeops_torch_trafo
#'
#' @description
#' This `PipeOp` represents simple preprocessing transformations of torch tensors.
#' These can be used in two situations:
#'
#' 1. To preprocess a task, which works analogous to standard preprocessing PipeOps like [`PipeOpPCA`].
#'    Because the [`lazy_tensor()`] does not make any assumptions on how the data is stored, the transformation is
#'    applied lazily, i.e. when [`materialize()`] is called.
#'    During trainig of a learner, this transformation will be a applied during data-loading on the CPU.
#'
#' 2. To add a preprocessing step in an [`nn_graph()`] that is being built up in a [`ModelDescriptor`].
#'    In this case, the transformation is applied during the forward pass of the model, i.e. the tensor is then
#'    also on the specified device.
#'
#' Currently the `PipeOp` must have exactly one inut and one output.
#'
#' @section Inheriting:
#' You need to:
#' * Initialize the `fn` argument. This function should take one torch tensor as input and return a torch tensor.
#'   Additional parameters that are passed to the function can be specified via the parameter set.
#'   This function needs to be a simple, stateless function, see section *Internals* for more information.
#' * In case the transformation changes the tensor shape you must provide a private `.shapes_out()` method like
#'   for [`PipeOpTorch`].
#'
#' @section Input and Output Channels:
#' During *training*, all inputs and outputs are of class [`Task`] or [`ModelDescriptor`].
#' During *prediction*, all inputs and outputs are of class [`Task`] or [`ModelDescriptor`].
#'
#' @template pipeop_torch_state_default
#' @section Parameters:
#' * `augment` :: `logical(1)`\cr
#'   This parameter is only present when the `PipeOp` does not modify the input shape.
#'   Whether the transformation is applied only during training (`TRUE`) or also during prediction (also includes
#'   validation; `FALSE`).
#'   This parameter is initalized to `FALSE`.
#'
#' Additional parameters can be specified by the class.
#'
#' @section Internals:
#'
#' Applied to a **Task**:
#'
#' When this PipeOp is used for preprocessing, it creates a [`PipeOpModule`] from the function `fn` (additionally
#' passing the `param_vals` if there are any) and then adds it to the preprocessing graph that is part of the
#' [`DataDescriptor`] contained in the [`lazy_tensor`] column that is being preprocessed.
#' When the outpuf of this pipeop is then preprocessed by a different `PipeOpTorchTrafo` a deep clone of the
#' preprocessing graph is done. However, this deep clone does not clone the environment of the
#' function or its attributes in case they have a state (as e.g. in [`nn_module()`]s).
#' When setting the parameter `augment` this meanst that the preprcessing
#'
#' Applied to a **ModelDescriptor**
#'
#'
#'
#'
#' @template param_id
#' @template param_param_vals
#' @template param_param_set
#' @param packages (`character()`)\cr
#'   The packages the function depends on.
#' @param fn (`function()`)\cr
#'   A function that will be applied to a (lazy) tensor.
#'   Additional arguments can be passed as parameters.
#'   During actual preprocessing the (lazy) tensor will be passed by position.
#'   The transformation is always applied to a whole batch of tensors, i.e. the first dimension is the batch dimension.
#'
#' @export
PipeOpTorchTrafo = R6Class("PipeOpTorchTrafo",
  inherit = PipeOpTorch,
  public = list(
    #' @description Creates a new instance of this [R6][R6::R6Class] class.
    initialize = function(fn, id = "torch_trafo", param_vals = list(), param_set = ps(), packages = character(0)) {
      private$.fn = assert_function(fn, null.ok = FALSE)

      if ("augment" %in% param_set$ids()) {
        stopf("Parameter name 'augment' is reserved and cannot be used.")
      }

      if (is.null(private$.shapes_out)) {
        param_set$add(ps(
          augment = p_lgl(tags = c("train", "required"))
        ))
        param_set$set_values(augment = FALSE)
      }

      super$initialize(
        id = id,
        inname = "input",
        outname = "output",
        param_vals = param_vals,
        param_set = param_set,
        packages = packages,
        module_generator = NULL
      )

    }
  ),
  private = list(
    .fn = NULL,
    .make_module = function(shapes_in, param_vals, task) {
      # this function is only called when the input is the ModelDescriptor
      augment = param_vals$augment
      param_vals$augment = NULL
      trafo = private$.fn
      fn = if (length(param_vals)) {
        crate(function(x) {
          invoke(.f = trafo, x, .args = param_vals)
        }, param_vals, trafo, .parent = topenv())
      } else {
        trafo
      }

      # augment can be NULL or logical(1)
      if (!isTRUE(augment)) {
        return(fn)
      }

      nn_module(self$id,
        initialize = function(fn) {
          self$fn = fn
        },
        forward = function(x) {
          if (self$training) {
            self$fn(x)
          } else {
            x
          }
        }
      )(fn)
    },
    .additional_phash_input = function() {
      list(self$param_set$ids(), private$.fn, self$packages)
    }
  )
)

pipeop_torch_trafo = function(fn, id, param_set = NULL) {

}

#' @include zzz.R
register_po("torch_trafo", PipeOpTorchTrafo)

